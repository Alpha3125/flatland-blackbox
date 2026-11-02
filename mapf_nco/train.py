"""Training with a differentiable planner (paper Sec. 3.4, Algorithm 1).

The downstream planner is PP (the paper) or CBS with learned costs
(``planner="cbs"``, see mapf_nco.solver_cbs.WeightedCBS); the black-box
gradient is the same for both. "PP" below stands for either.

Per mini-batch of B scenarios:
    w_j      = N_theta(x_j)                 predict edge costs
    y_hat_j  = PP(w_j)                      forward pass
    w'_j     = w_j + lambda * dL/dy_hat_j    perturb
    y_lam_j  = PP(w'_j)                     perturbed pass
    grad_j   = -(y_hat_j - y_lam_j)/lambda   surrogate gradient
then back-propagate {grad_j} through N_theta and take an Adam step.

Model selection: after every epoch we compute the validation delay (PP's
sum-of-costs minus the collision-free lower bound) and keep the checkpoint with
the lowest value. Epoch 0 (the untrained model, which equals original PP) is a
valid checkpoint, so the selected model is never worse than PP on validation.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

from mapf_nco.blackbox import blackbox_step, makespan_loss, usage_loss
from mapf_nco.grid import Grid
from mapf_nco.models import build_model, make_batch
from mapf_nco.scenarios import dataset_path, load_dataset, train_val_split
from mapf_nco.metrics import edge_usage
from mapf_nco.paths import checkpoint_path, tensorboard_dir, training_dir
from mapf_nco.planners import MODEL_IDS, model_id
from mapf_nco.solver_cbs import WeightedCBS
from mapf_nco.sequential import METHODS, SEQ_DIM, model_spec, seq_plan_task, seq_train_task
from mapf_nco.solver_pp import NoSolutionError, PrioritizedPlanner
from mapf_nco.tb import TB
from mapf_nco.viz import plot_training_curves

# ----------------------------------------------------------- worker processes
_PLANNERS = {}


def _init_worker(map_path, w_min, cbs_time_limit=10.0, single_thread=True):
    if single_thread:
        torch.set_num_threads(1)
    grid = Grid.from_file(map_path)
    _PLANNERS["pp"] = PrioritizedPlanner(grid, w_min=w_min)
    _PLANNERS["cbs"] = WeightedCBS(grid, w_min=w_min, time_limit=cbs_time_limit)


def _step_task(args):
    starts, goals, order, w, y_star, lam, reduction, obj, pl = args
    return blackbox_step(_PLANNERS[pl], starts, goals, order, w, y_star, lam, reduction, **obj)


def _loss_task(args):
    starts, goals, order, w, y_star, reduction, obj, pl = args
    planner = _PLANNERS[pl]
    try:
        paths = planner.solve(starts, goals, order, w)
    except NoSolutionError:
        return None
    return objective_loss(planner.grid, paths, y_star, reduction, **obj)


def objective_loss(grid, paths, y_star, reduction, objective="usage", tau=1.0, beta=1.0):
    loss = 0.0
    if objective in ("usage", "usage+makespan"):
        loss += usage_loss(edge_usage(grid, paths), y_star, reduction)[0]
    if objective in ("makespan", "usage+makespan"):
        loss += (beta if objective == "usage+makespan" else 1.0) * makespan_loss(paths, tau)[0]
    return loss


def _plan_task(args):
    """(sum of costs, makespan, seconds, CBS high-level nodes); -1 if it failed."""
    starts, goals, order, w, pl = args
    planner = _PLANNERS[pl]
    t0 = time.perf_counter()
    try:
        paths = planner.solve(starts, goals, order, w)
    except NoSolutionError:
        return -1, -1, time.perf_counter() - t0, getattr(planner, "last_nodes", 0)
    return (sum(len(p) - 1 for p in paths), max(len(p) - 1 for p in paths),
            time.perf_counter() - t0, getattr(planner, "last_nodes", 0))


class PPRunner:
    """Runs planner tasks (PP or CBS) either in-process or in a process pool."""

    def __init__(self, map_path, w_min, workers, cbs_time_limit=10.0):
        self.workers = workers
        _init_worker(map_path, w_min, cbs_time_limit, single_thread=False)  # in-process use
        self.pool = (
            ProcessPoolExecutor(workers, initializer=_init_worker,
                                initargs=(map_path, w_min, cbs_time_limit))
            if workers > 1 else None
        )

    def map(self, fn, tasks):
        if self.pool is None:
            return [fn(t) for t in tasks]
        return list(self.pool.map(fn, tasks, chunksize=max(1, len(tasks) // (4 * self.workers))))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


# --------------------------------------------------------------------- helpers
def predict_weights(model, scenarios, chunk=16, device="cpu"):
    """Edge costs for each scenario, as a float64 NumPy array (S, M)."""
    out = []
    model.eval()
    with torch.no_grad():
        for k in range(0, len(scenarios), chunk):
            w = model(make_batch(scenarios[k:k + chunk], device))
            out.append(w.detach().cpu().double().numpy())
    model.train()
    return np.concatenate(out, 0)


def _sequential_backward(model, batch_sc, results, device, chunk=32):
    """Back-propagate the per-agent gradients through every model call.

    Each (scenario, agent) state is re-run through the model with its saved
    reservation features; gradients are averaged over the scenarios of the
    batch and summed over agents (as in the static method).
    """
    states = []
    for sc, r in zip(batch_sc, results):
        if not r["ok"]:
            continue
        for x, g in zip(r["X"], r["G"]):
            if np.any(g):
                states.append((sc, x, g))
    scale = 1.0 / max(1, len(results))
    for k in range(0, len(states), chunk):
        part = states[k:k + chunk]
        w = model(make_batch([s for s, _, _ in part], device, seq=np.stack([x for _, x, _ in part])))
        grads = np.stack([g for _, _, g in part]) * scale
        w.backward(torch.from_numpy(grads).to(w.dtype).to(device))


def plan_delays(runner, grid, scenarios, weights, fallback=None, planner="pp"):
    """Delay and makespan of the planner under ``weights`` for each scenario.

    If learned weights make the planner fail (PP: no path; CBS: time limit), the
    scenario falls back to ``fallback`` = (soc, makespan) arrays and is counted
    as a failure. Returns (delays, fails, times, makespans, cbs_nodes).
    """
    if isinstance(weights, tuple) and weights and weights[0] == "seq":
        _, spec, map_path, w_min = weights
        res = runner.map(seq_plan_task, [(spec, map_path, w_min, s.starts, s.goals, s.order)
                                         for s in scenarios])
    else:
        tasks = [(s.starts, s.goals, s.order, None if weights is None else weights[i], planner)
                 for i, s in enumerate(scenarios)]
        res = runner.map(_plan_task, tasks)
    delays, fails, times, spans, nodes = [], 0, [], [], []
    for i, (s, r) in enumerate(zip(scenarios, res)):
        soc, span, t = r[:3]
        times.append(t)
        nodes.append(r[3] if len(r) > 3 else 0)
        if soc < 0:
            fails += 1
            if fallback is None:
                raise NoSolutionError(i, f"{planner.upper()} failed with the original costs")
            soc, span = fallback[0][i], fallback[1][i]
        delays.append(soc - s.lower_bound(grid))
        spans.append(span)
    return (np.array(delays, dtype=np.float64), fails, np.array(times),
            np.array(spans, dtype=np.float64), np.array(nodes, dtype=np.float64))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------- training
def train(cfg, n, kind, workers=1, log=print, planner="pp"):
    """Train one model (``kind`` edge/gnn) for one downstream planner (pp/cbs).

    Returns the checkpoint path. Writes (replacing earlier runs of the same model):
      checkpoints/<id>.pt, training/{curves.png,history.csv,history.json}, tensorboard/<id>/
    with <id> = edge, gnn (PP) or edge_cbs, gnn_cbs (CBS).
    """
    if planner not in ("pp", "cbs"):
        raise ValueError(f"planner must be pp or cbs, not {planner!r}")
    mid = model_id(kind, planner)
    PL = planner.upper()
    if planner == "cbs" and cfg.get("method", "static") != "static":
        log(f"  note: method {cfg['method']} recomputes costs between PP's agents; CBS plans "
            "all agents together, so the CBS models use the static method")
        cfg = dict(cfg, method="static")
    set_seed(cfg.get("seed", 0))
    grid = Grid.from_file(cfg["map_path"])
    data, info = load_dataset(dataset_path(cfg["data_dir"], grid.name, n, "train"), grid)
    ename = "EECBS" if info.get("expert", "eecbs") == "eecbs" else "CBS"
    train_set, val_set = train_val_split(data, cfg["val_fraction"], cfg.get("seed", 0))
    objective = cfg.get("objective", "usage")
    obj = {"objective": objective, "tau": float(cfg.get("makespan_tau", 1.0)),
           "beta": float(cfg.get("makespan_weight", 0.001))}
    metric = "delay" if objective == "usage" else "makespan"  # what picks the best epoch
    method = cfg.get("method", "static")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "seq_teacher" and objective != "usage":
        log("  note: seq_teacher learns from the expert's paths; the objective only affects "
            "which epoch is selected")
    log(f"[train] n={n} model={kind}, planner {PL}: {len(train_set)} train / {len(val_set)} val scenarios, "
        f"{cfg['epochs']} epochs, batch {cfg['batch_size']}, lr {cfg['lr']}, lambda {cfg['lam']}, "
        f"method {method}, objective {objective}" + ("" if objective == "usage" else
                                    f" (tau {obj['tau']}" + (f", weight {obj['beta']})"
                                    if objective == "usage+makespan" else ")")))

    device = cfg.get("device", "cpu")
    model_kwargs = dict(cfg.get("gnn", {})) if kind == "gnn" else {}
    if kind == "gnn":
        model_kwargs.setdefault("max_agents", max(128, n))
        if method != "static":
            model_kwargs["seq_dim"] = SEQ_DIM  # sees the reservation state before each agent
    model = build_model(kind, grid, **model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lam, reduction, B = cfg["lam"], cfg.get("loss_reduction", "mean"), cfg["batch_size"]
    y_star = [s.expert_usage(grid) for s in train_set]

    ckpt = checkpoint_path(cfg, grid.name, n, mid)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    tdir = training_dir(cfg, grid.name, n)
    os.makedirs(tdir, exist_ok=True)
    tb = TB(tensorboard_dir(cfg, grid.name, n, mid), log)
    tb.text("config", "\n".join(f"    {k}: {v}" for k, v in sorted(cfg.items())))

    runner = PPRunner(cfg["map_path"], cfg["w_min"], workers,
                      float(cfg.get("cbs_train_time_limit", 10)))
    history, batch_losses = [], []
    try:
        # The planner with its original costs on validation: fallback for failures
        # and reference line. Original CBS falls back to the expert if it times out.
        expert_spans = [max(len(p) - 1 for p in s.expert_paths) for s in val_set]
        expert_fb = None
        if planner == "cbs":
            expert_fb = (np.array([sum(len(p) - 1 for p in s.expert_paths) for s in val_set]),
                         np.array(expert_spans))
        base_delays, base_to, _, base_spans, base_nodes = plan_delays(
            runner, grid, val_set, None, expert_fb, planner)
        if base_to:
            log(f"  original CBS timed out on {base_to}/{len(val_set)} validation scenarios; "
                "the expert plan is used for those")
        base_soc = base_delays + np.array([s.lower_bound(grid) for s in val_set])
        refs = {"metric": metric, "expert_name": ename,
                "orig_pp": float(base_delays.mean() if metric == "delay" else base_spans.mean()),
                "expert": float(np.mean([s.expert_delay(grid) for s in val_set])
                                if metric == "delay" else np.mean(expert_spans)),
                "objective": objective, "tau": obj["tau"], "beta": obj["beta"],
                "method": method, "planner": planner}

        def validate():
            if method == "static":
                weights = predict_weights(model, val_set, device=device)
            else:
                weights = ("seq", model_spec(kind, model_kwargs, model), cfg["map_path"],
                           cfg["w_min"])
            d, fails, _, sp, nodes = plan_delays(runner, grid, val_set, weights,
                                                 (base_soc, base_spans), planner)
            return float(d.mean()), float(sp.mean()), fails, float(nodes.mean())

        def score(d, sp, nodes):
            # Best epoch: lowest validation delay (or makespan). For CBS, ties are
            # broken by the search effort (high-level nodes): CBS is already optimal
            # with unit costs, so learned costs can mostly make it faster.
            v = round(d if metric == "delay" else sp, 6)
            return (v, nodes) if planner == "cbs" else (v,)

        def seq_tasks(scen_idx):
            spec = model_spec(kind, model_kwargs, model)
            return [(spec, cfg["map_path"], cfg["w_min"], train_set[i].starts,
                     train_set[i].goals, train_set[i].order, train_set[i].expert_paths,
                     y_star[i], method, lam, reduction, obj) for i in scen_idx]

        def train_loss_now():
            if method != "static":
                res = runner.map(seq_train_task, seq_tasks(range(len(train_set))))
                vals = [r["loss"] for r in res if r["ok"]]
                return float(np.mean(vals)) if vals else None
            w_all = predict_weights(model, train_set, device=device)
            tasks = [(s.starts, s.goals, s.order, w_all[i], y_star[i], reduction, obj, planner)
                     for i, s in enumerate(train_set)]
            vals = [v for v in runner.map(_loss_task, tasks) if v is not None]
            return float(np.mean(vals)) if vals else None

        loss0 = train_loss_now()
        d0, s0, fails, nd0 = validate()
        best = score(d0, s0, nd0)
        best_epoch = 0
        save_checkpoint(ckpt, model, kind, model_kwargs, n, grid, 0, d0, cfg, s0, planner)
        history.append({"epoch": 0, "train_loss": loss0, "val_delay": d0, "val_makespan": s0,
                        "val_failures": fails})
        if planner == "cbs":
            history[-1]["val_cbs_nodes"] = nd0
            refs["orig_nodes"] = float(base_nodes.mean())
        _log_epoch(tb, model, kind, history[-1], refs)
        log(f"  epoch   0  loss {_fmt(loss0)}  val delay {d0:7.2f}  makespan {s0:6.2f}"
            + (f"  CBS nodes {nd0:7.1f}" if planner == "cbs" else "")
            + f"   (original {PL} {refs['orig_pp']:.2f}, {ename} {refs['expert']:.2f} {metric}"
            + (f", original CBS nodes {refs['orig_nodes']:.1f})" if planner == "cbs" else ")"))

        rng = random.Random(cfg.get("seed", 0))
        step = 0
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            idx = list(range(len(train_set)))
            rng.shuffle(idx)
            losses, n_fail, n_zero, n_exact = [], 0, 0, 0
            bar = tqdm(range(0, len(idx), B), desc=f"epoch {epoch}/{cfg['epochs']}",
                       unit="batch", leave=False)
            for k in bar:
                bidx = idx[k:k + B]
                batch_sc = [train_set[i] for i in bidx]
                opt.zero_grad()
                if method == "static":
                    w = model(make_batch(batch_sc, device))  # (b, M)
                    w_np = w.detach().cpu().double().numpy()
                    tasks = [(s.starts, s.goals, s.order, w_np[j], y_star[i], lam, reduction,
                              obj, planner) for j, (i, s) in enumerate(zip(bidx, batch_sc))]
                    results = runner.map(_step_task, tasks)
                    grads = np.stack([r.grad for r in results]) / len(results)
                    b_losses = [r.loss for r in results if r.ok]
                    n_fail += sum(not r.ok for r in results)
                    n_zero += sum(not np.any(r.grad) for r in results)
                    w.backward(torch.from_numpy(grads).to(w.dtype).to(device))
                else:
                    results = runner.map(seq_train_task, seq_tasks(bidx))
                    b_losses = [r["loss"] for r in results if r["ok"]]
                    n_fail += sum(not r["ok"] for r in results)
                    n_zero += sum((not r["ok"]) or not np.any(r["G"]) for r in results)
                    n_exact += sum(r["exact"] for r in results)
                    _sequential_backward(model, batch_sc, results, device)
                losses += b_losses
                opt.step()
                step += 1
                if b_losses:
                    bar.set_postfix(loss=f"{np.mean(losses):.5f}")
                    batch_losses.append((epoch - 1 + (k + len(bidx)) / len(idx), float(np.mean(b_losses))))
                    tb.scalar("train/batch_loss", batch_losses[-1][1], step)
            bar.close()
            vd, vs, vfails, vnodes = validate()
            val = score(vd, vs, vnodes)
            improved = val < best
            if improved:
                best, best_epoch = val, epoch
                save_checkpoint(ckpt, model, kind, model_kwargs, n, grid, epoch, vd, cfg, vs,
                                planner)
            history.append({
                "epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
                "val_delay": vd, "val_makespan": vs, "val_failures": vfails,
                "train_pp_failures": n_fail, "zero_grad_scenarios": n_zero,
                "seconds": time.time() - t0,
            })
            if planner == "cbs":
                history[-1]["val_cbs_nodes"] = vnodes
            if method == "seq_teacher":
                history[-1]["teacher_exact_steps"] = n_exact / max(1, len(idx) * n)
            _log_epoch(tb, model, kind, history[-1], refs)
            log(f"  epoch {epoch:3d}  loss {_fmt(history[-1]['train_loss'])}  val delay {vd:7.2f}  "
                f"makespan {vs:6.2f}" + (f"  CBS nodes {vnodes:7.1f}" if planner == "cbs" else "")
                + f"{'  *best' if improved else '       '}  "
                f"[{PL} fails val {vfails}/train {n_fail}, zero-grad {n_zero}/{len(idx)}, "
                + (f"expert steps reproduced {100 * n_exact / max(1, len(idx) * n):.0f}%, "
                   if method == "seq_teacher" else "")
                + f"{time.time() - t0:.0f}s]")
    finally:
        runner.close()

    runs = _update_runs(tdir, mid, {"history": history, "batch_losses": batch_losses,
                                     "refs": refs, "best_epoch": best_epoch,
                                     "reduction": reduction})
    curves = plot_training_curves(runs, n, os.path.join(tdir, "curves.png"))
    tb.image_file("curves", curves)
    tb.close()

    final = history[-1]
    log(f"  done: final loss {_fmt(final['train_loss'])}, best val {metric} {best[0]:.2f} at epoch "
        f"{best_epoch} (original {PL} {refs['orig_pp']:.2f}, {ename} {refs['expert']:.2f})")
    log(f"  model:       {ckpt}")
    log(f"  loss graph:  {curves}")
    log(f"  tensorboard: tensorboard --logdir {os.path.dirname(tensorboard_dir(cfg, grid.name, n, mid))}")
    return ckpt


def _fmt(v):
    return "   n/a  " if v is None else f"{v:.5f}"


def _log_epoch(tb, model, kind, h, refs):
    e = h["epoch"]
    tb.scalar("train/loss", h.get("train_loss"), e)
    tb.scalar("val/delay", h["val_delay"], e)
    tb.scalar("val/makespan", h.get("val_makespan"), e)
    tb.scalar(f"val/{refs['metric']}_original_pp", refs["orig_pp"], e)
    tb.scalar(f"val/{refs['metric']}_expert", refs["expert"], e)
    tb.scalar("val/planner_failures", h.get("val_failures"), e)
    tb.scalar("train/planner_failures", h.get("train_pp_failures"), e)
    tb.scalar("val/cbs_nodes", h.get("val_cbs_nodes"), e)
    tb.scalar("train/zero_grad_scenarios", h.get("zero_grad_scenarios"), e)
    tb.scalar("time/epoch_seconds", h.get("seconds"), e)
    if kind == "edge":
        tb.histogram("weights/edge_costs", model.edge_weights.detach().cpu().numpy(), e)


def _update_runs(out_dir, kind, run):
    """Store this model's run next to the other model's; rewrite the shared files.

    history.json holds every model's run (used to redraw the combined plot);
    history.csv has one row per (model, epoch). Retraining a model replaces only
    that model's entries.
    """
    path = os.path.join(out_dir, "history.json")
    runs = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                runs = json.load(f)
            if not isinstance(runs, dict) or any(not isinstance(v, dict) for v in runs.values()):
                runs = {}  # old per-model format
        except (OSError, ValueError):
            runs = {}
    runs[kind] = run
    runs = {k: runs[k] for k in MODEL_IDS if k in runs}
    with open(path, "w") as f:
        json.dump(runs, f, indent=1)
    keys = ["model", "epoch", "train_loss", "val_delay", "val_makespan", "val_failures",
            "train_pp_failures", "zero_grad_scenarios", "teacher_exact_steps", "val_cbs_nodes",
            "seconds"]
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for k, r in runs.items():
            w.writerows({"model": k, **h} for h in r["history"])
    return runs


def save_checkpoint(path, model, kind, model_kwargs, n, grid, epoch, val_delay, cfg,
                    val_makespan=None, planner="pp"):
    torch.save({
        "kind": kind, "planner": planner, "model_kwargs": model_kwargs, "state_dict": model.state_dict(),
        "num_agents": n, "map": grid.name, "num_edges": grid.num_edges,
        "epoch": epoch, "val_delay": val_delay, "val_makespan": val_makespan,
        "config": {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, list, dict))},
    }, path)


def load_model(path, grid, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["num_edges"] != grid.num_edges:
        raise ValueError(f"{path} was trained on a different map ({ck['map']})")
    model = build_model(ck["kind"], grid, **ck["model_kwargs"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
