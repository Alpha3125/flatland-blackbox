"""Training with a differentiable planner (paper Sec. 3.4, Algorithm 1).

The downstream planner is PP (the paper) or CBS with learned costs
(``planner="cbs"``, see mapf_nco.solver_cbs.WeightedCBS); the black-box
gradient is the same for both. "PP" below stands for either.

The model (``kind``) produces one cost per directed edge (incl. waits), shared
by every scenario: the paper's Edge model, the potential model (PotP), or the
Edge model started from a trained potential model's costs (potential_edge).
Per mini-batch of B scenarios:
    w_j      = w(theta)                     the current edge costs
    y_hat_j  = PP(w_j)                      forward pass
    w'_j     = w_j + lambda * dL/dy_hat_j    perturb
    y_lam_j  = PP(w'_j)                     perturbed pass
    grad_j   = -(y_hat_j - y_lam_j)/lambda   surrogate gradient
then average {grad_j} over the batch, back-propagate it to the model's
parameters theta and take an Adam step.

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
from mapf_nco.planners import KIND_LABELS, MODEL_IDS, model_id
from mapf_nco.potential import describe
from mapf_nco.solver_cbs import WeightedCBS
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


def plan_delays(runner, grid, scenarios, weights, fallback=None, planner="pp"):
    """Delay and makespan of the planner under ``weights`` for each scenario.

    If learned weights make the planner fail (PP: no path; CBS: time limit), the
    scenario falls back to ``fallback`` = (soc, makespan) arrays and is counted
    as a failure. Returns (delays, fails, times, makespans, cbs_nodes).
    """
    tasks = [(s.starts, s.goals, s.order, None if weights is None else weights[i], planner)
             for i, s in enumerate(scenarios)]
    res = runner.map(_plan_task, tasks)
    delays, fails, times, spans, nodes = [], 0, [], [], []
    for i, (s, (soc, span, t, nd)) in enumerate(zip(scenarios, res)):
        times.append(t)
        nodes.append(nd)
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
def train(cfg, n, planner="pp", workers=1, log=print, kind="edge"):
    """Train one model (``kind``: edge / potential / potential_edge) for one
    downstream planner (pp / cbs). Returns the checkpoint path.

    potential_edge starts from the trained potential model of the same planner
    (train ``kind="potential"`` first; the menus and commands do this).

    Writes (replacing earlier runs of the same model):
      checkpoints/<id>.pt, training/{curves.png,history.csv,history.json}, tensorboard/<id>/
    with <id> = <kind> (PP) or <kind>_cbs (CBS).
    """
    if planner not in ("pp", "cbs"):
        raise ValueError(f"planner must be pp or cbs, not {planner!r}")
    if kind not in KIND_LABELS:
        raise ValueError(f"model must be one of {', '.join(KIND_LABELS)}, not {kind!r}")
    mid = model_id(planner, kind)
    PL = planner.upper()
    set_seed(cfg.get("seed", 0))
    grid = Grid.from_file(cfg["map_path"])
    data, info = load_dataset(dataset_path(cfg["data_dir"], grid.name, n, "train"), grid)
    ename = "EECBS" if info.get("expert", "eecbs") == "eecbs" else "CBS"
    train_set, val_set = train_val_split(data, cfg["val_fraction"], cfg.get("seed", 0))
    objective = cfg.get("objective", "usage")
    obj = {"objective": objective, "tau": float(cfg.get("makespan_tau", 1.0)),
           "beta": float(cfg.get("makespan_weight", 0.001))}
    metric = "delay" if objective == "usage" else "makespan"  # what picks the best epoch
    log(f"[train] n={n} {KIND_LABELS[kind]} model, planner {PL}: {len(train_set)} train / "
        f"{len(val_set)} val "
        f"scenarios, {cfg['epochs']} epochs, batch {cfg['batch_size']}, lr {cfg['lr']}, "
        f"lambda {cfg['lam']}, objective {objective}" + ("" if objective == "usage" else
                                    f" (tau {obj['tau']}" + (f", weight {obj['beta']})"
                                    if objective == "usage+makespan" else ")")))

    device = cfg.get("device", "cpu")
    model_kwargs = {}  # what load_model needs to rebuild the model
    if kind == "potential":
        model_kwargs = {"init": dict(cfg.get("potential_init", {}))}
        model = build_model(kind, grid, **model_kwargs).to(device)
        log(f"  start: {describe(grid, model.theta.detach().cpu().numpy())}")
    elif kind == "potential_edge":
        pot_ckpt = checkpoint_path(cfg, grid.name, n, model_id(planner, "potential"))
        if not os.path.exists(pot_ckpt):
            raise FileNotFoundError(f"{pot_ckpt} not found: potential_edge starts from the trained "
                                    "potential model, so train model 'potential' first")
        pot, _ = load_model(pot_ckpt, grid, device)
        with torch.no_grad():
            w0 = np.maximum(pot.costs().detach().cpu().double().numpy(), cfg["w_min"])
        model = build_model(kind, grid, init=w0).to(device)
        log(f"  start: the trained potential model's costs ({pot_ckpt})")
    else:
        model = build_model(kind, grid).to(device)
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
                "planner": planner, "kind": kind}

        def validate():
            weights = predict_weights(model, val_set, device=device)
            d, fails, _, sp, nodes = plan_delays(runner, grid, val_set, weights,
                                                 (base_soc, base_spans), planner)
            return float(d.mean()), float(sp.mean()), fails, float(nodes.mean())

        def score(d, sp, nodes):
            # Best epoch: lowest validation delay (or makespan). For CBS, ties are
            # broken by the search effort (high-level nodes): CBS is already optimal
            # with unit costs, so learned costs can mostly make it faster.
            v = round(d if metric == "delay" else sp, 6)
            return (v, nodes) if planner == "cbs" else (v,)

        def train_loss_now():
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
            losses, n_fail, n_zero = [], 0, 0
            bar = tqdm(range(0, len(idx), B), desc=f"epoch {epoch}/{cfg['epochs']}",
                       unit="batch", leave=False)
            for k in bar:
                bidx = idx[k:k + B]
                batch_sc = [train_set[i] for i in bidx]
                opt.zero_grad()
                w = model(make_batch(batch_sc, device))  # (b, M): the same costs for each
                w_np = w.detach().cpu().double().numpy()
                tasks = [(s.starts, s.goals, s.order, w_np[j], y_star[i], lam, reduction,
                          obj, planner) for j, (i, s) in enumerate(zip(bidx, batch_sc))]
                results = runner.map(_step_task, tasks)
                grads = np.stack([r.grad for r in results]) / len(results)
                b_losses = [r.loss for r in results if r.ok]
                n_fail += sum(not r.ok for r in results)
                n_zero += sum(not np.any(r.grad) for r in results)
                w.backward(torch.from_numpy(grads).to(w.dtype).to(device))
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
            _log_epoch(tb, model, kind, history[-1], refs)
            log(f"  epoch {epoch:3d}  loss {_fmt(history[-1]['train_loss'])}  val delay {vd:7.2f}  "
                f"makespan {vs:6.2f}" + (f"  CBS nodes {vnodes:7.1f}" if planner == "cbs" else "")
                + f"{'  *best' if improved else '       '}  "
                f"[{PL} fails val {vfails}/train {n_fail}, zero-grad {n_zero}/{len(idx)}, "
                f"{time.time() - t0:.0f}s]")
            if kind == "potential":
                log(f"             {describe(grid, model.theta.detach().cpu().numpy())}")
    finally:
        runner.close()

    runs = _update_runs(tdir, mid, {"history": history, "batch_losses": batch_losses,
                                     "refs": refs, "best_epoch": best_epoch,
                                     "reduction": reduction})
    curves = plot_training_curves(runs, n, os.path.join(tdir, "curves.png"), current=mid)
    tb.image_file("curves", curves)
    tb.close()

    final = history[-1]
    log(f"  done: final loss {_fmt(final['train_loss'])}, best val {metric} {best[0]:.2f} at epoch "
        f"{best_epoch} (original {PL} {refs['orig_pp']:.2f}, {ename} {refs['expert']:.2f})")
    if best[0] > refs["orig_pp"] + 1e-9:
        log(f"  note: the selected model is worse than the original {PL} on validation "
            f"({best[0]:.2f} vs {refs['orig_pp']:.2f} {metric})"
            + ("; its starting strengths (potential_init) don't help on this data"
               if kind != "edge" else ""))
    if kind == "potential":
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        log(f"  learned strengths (epoch {best_epoch}): "
            f"{describe(grid, np.asarray(ck['state_dict']['theta'], dtype=float))}")
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
    with torch.no_grad():
        tb.histogram("weights/edge_costs", model.costs().detach().cpu().numpy(), e)
    if kind == "potential":
        from mapf_nco.potential import STRENGTHS

        theta = model.theta.detach().cpu().numpy()
        for i, name in enumerate(STRENGTHS):
            tb.scalar(f"potential/{name}", float(theta[i]), e)


def _update_runs(out_dir, kind, run):
    """Store this model's run next to the other planner's; rewrite the shared files.

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
            "train_pp_failures", "zero_grad_scenarios", "val_cbs_nodes", "seconds"]
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
    model = build_model(ck["kind"], grid, **ck.get("model_kwargs", {})).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
