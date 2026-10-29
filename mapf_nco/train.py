"""Training with differentiable PP (paper Sec. 3.4, Algorithm 1).

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
from mapf_nco.paths import checkpoint_path, fresh_dir, tensorboard_dir, training_dir
from mapf_nco.solver_pp import NoSolutionError, PrioritizedPlanner
from mapf_nco.tb import TB
from mapf_nco.viz import plot_training_curves

# ----------------------------------------------------------- worker processes
_PLANNER = None


def _init_worker(map_path, w_min, single_thread=True):
    global _PLANNER
    if single_thread:
        torch.set_num_threads(1)
    _PLANNER = PrioritizedPlanner(Grid.from_file(map_path), w_min=w_min)


def _step_task(args):
    starts, goals, order, w, y_star, lam, reduction, obj = args
    return blackbox_step(_PLANNER, starts, goals, order, w, y_star, lam, reduction, **obj)


def _loss_task(args):
    starts, goals, order, w, y_star, reduction, obj = args
    try:
        paths = _PLANNER.solve(starts, goals, order, w)
    except NoSolutionError:
        return None
    return objective_loss(_PLANNER.grid, paths, y_star, reduction, **obj)


def objective_loss(grid, paths, y_star, reduction, objective="usage", tau=1.0, beta=1.0):
    loss = 0.0
    if objective in ("usage", "usage+makespan"):
        loss += usage_loss(edge_usage(grid, paths), y_star, reduction)[0]
    if objective in ("makespan", "usage+makespan"):
        loss += (beta if objective == "usage+makespan" else 1.0) * makespan_loss(paths, tau)[0]
    return loss


def _plan_task(args):
    starts, goals, order, w = args
    t0 = time.perf_counter()
    try:
        paths = _PLANNER.solve(starts, goals, order, w)
    except NoSolutionError:
        return -1, -1, time.perf_counter() - t0
    return (sum(len(p) - 1 for p in paths), max(len(p) - 1 for p in paths),
            time.perf_counter() - t0)


class PPRunner:
    """Runs PP tasks either in-process or in a process pool."""

    def __init__(self, map_path, w_min, workers):
        self.workers = workers
        _init_worker(map_path, w_min, single_thread=False)  # in-process use
        self.pool = (
            ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(map_path, w_min))
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


def plan_delays(runner, grid, scenarios, weights, fallback=None):
    """Delay and makespan of PP under ``weights`` for each scenario.

    If learned weights make PP fail, the scenario falls back to the original PP
    plan (``fallback`` = (soc, makespan) arrays) and is counted as a failure.
    Returns (delays, fails, times, makespans).
    """
    tasks = [(s.starts, s.goals, s.order, None if weights is None else weights[i])
             for i, s in enumerate(scenarios)]
    res = runner.map(_plan_task, tasks)
    delays, fails, times, spans = [], 0, [], []
    for i, (s, (soc, span, t)) in enumerate(zip(scenarios, res)):
        times.append(t)
        if soc < 0:
            fails += 1
            if fallback is None:
                raise NoSolutionError(i, "PP failed with the original costs")
            soc, span = fallback[0][i], fallback[1][i]
        delays.append(soc - s.lower_bound(grid))
        spans.append(span)
    return (np.array(delays, dtype=np.float64), fails, np.array(times),
            np.array(spans, dtype=np.float64))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------- training
def train(cfg, n, kind, workers=1, log=print):
    """Train one model from scratch. Returns the checkpoint path.

    Writes (replacing earlier runs of the same model):
      checkpoints/<kind>.pt, training/<kind>/{curves.png,history.csv,history.json},
      tensorboard/<kind>/
    """
    set_seed(cfg.get("seed", 0))
    grid = Grid.from_file(cfg["map_path"])
    data, info = load_dataset(dataset_path(cfg["data_dir"], grid.name, n, "train"), grid)
    ename = "EECBS" if info.get("expert", "eecbs") == "eecbs" else "CBS"
    train_set, val_set = train_val_split(data, cfg["val_fraction"], cfg.get("seed", 0))
    objective = cfg.get("objective", "usage")
    obj = {"objective": objective, "tau": float(cfg.get("makespan_tau", 1.0)),
           "beta": float(cfg.get("makespan_weight", 0.001))}
    metric = "delay" if objective == "usage" else "makespan"  # what picks the best epoch
    log(f"[train] n={n} model={kind}: {len(train_set)} train / {len(val_set)} val scenarios, "
        f"{cfg['epochs']} epochs, batch {cfg['batch_size']}, lr {cfg['lr']}, lambda {cfg['lam']}, "
        f"objective {objective}" + ("" if objective == "usage" else
                                    f" (tau {obj['tau']}" + (f", weight {obj['beta']})"
                                    if objective == "usage+makespan" else ")")))

    device = cfg.get("device", "cpu")
    model_kwargs = dict(cfg.get("gnn", {})) if kind == "gnn" else {}
    if kind == "gnn":
        model_kwargs.setdefault("max_agents", max(128, n))
    model = build_model(kind, grid, **model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lam, reduction, B = cfg["lam"], cfg.get("loss_reduction", "mean"), cfg["batch_size"]
    y_star = [s.expert_usage(grid) for s in train_set]

    ckpt = checkpoint_path(cfg, grid.name, n, kind)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    tdir = fresh_dir(training_dir(cfg, grid.name, n, kind))
    tb = TB(tensorboard_dir(cfg, grid.name, n, kind), log)
    tb.text("config", "\n".join(f"    {k}: {v}" for k, v in sorted(cfg.items())))

    runner = PPRunner(cfg["map_path"], cfg["w_min"], workers)
    history, batch_losses = [], []
    try:
        # Original PP on validation: fallback for failures and reference line.
        base_delays, _, _, base_spans = plan_delays(runner, grid, val_set, None)
        base_soc = base_delays + np.array([s.lower_bound(grid) for s in val_set])
        expert_spans = [max(len(p) - 1 for p in s.expert_paths) for s in val_set]
        refs = {"metric": metric, "expert_name": ename,
                "orig_pp": float(base_delays.mean() if metric == "delay" else base_spans.mean()),
                "expert": float(np.mean([s.expert_delay(grid) for s in val_set])
                                if metric == "delay" else np.mean(expert_spans)),
                "objective": objective, "tau": obj["tau"], "beta": obj["beta"]}

        def validate():
            d, fails, _, sp = plan_delays(runner, grid, val_set,
                                          predict_weights(model, val_set, device=device),
                                          (base_soc, base_spans))
            return float(d.mean()), float(sp.mean()), fails

        def train_loss_now():
            w_all = predict_weights(model, train_set, device=device)
            tasks = [(s.starts, s.goals, s.order, w_all[i], y_star[i], reduction, obj)
                     for i, s in enumerate(train_set)]
            vals = [v for v in runner.map(_loss_task, tasks) if v is not None]
            return float(np.mean(vals)) if vals else None

        loss0 = train_loss_now()
        d0, s0, fails = validate()
        best = d0 if metric == "delay" else s0
        best_epoch = 0
        save_checkpoint(ckpt, model, kind, model_kwargs, n, grid, 0, d0, cfg, s0)
        history.append({"epoch": 0, "train_loss": loss0, "val_delay": d0, "val_makespan": s0,
                        "val_failures": fails})
        _log_epoch(tb, model, kind, history[-1], refs)
        log(f"  epoch   0  loss {_fmt(loss0)}  val delay {d0:7.2f}  makespan {s0:6.2f}   "
            f"(original PP {refs['orig_pp']:.2f}, {ename} {refs['expert']:.2f} {metric})")

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
                w = model(make_batch(batch_sc, device))  # (b, M)
                w_np = w.detach().cpu().double().numpy()
                tasks = [(s.starts, s.goals, s.order, w_np[j], y_star[i], lam, reduction, obj)
                         for j, (i, s) in enumerate(zip(bidx, batch_sc))]
                results = runner.map(_step_task, tasks)
                grads = np.stack([r.grad for r in results]) / len(results)
                b_losses = [r.loss for r in results if r.ok]
                n_fail += sum(not r.ok for r in results)
                n_zero += sum(not np.any(r.grad) for r in results)
                losses += b_losses
                w.backward(torch.from_numpy(grads).to(w.dtype).to(device))
                opt.step()
                step += 1
                if b_losses:
                    bar.set_postfix(loss=f"{np.mean(losses):.5f}")
                    batch_losses.append((epoch - 1 + (k + len(bidx)) / len(idx), float(np.mean(b_losses))))
                    tb.scalar("train/batch_loss", batch_losses[-1][1], step)
            bar.close()
            vd, vs, vfails = validate()
            val = vd if metric == "delay" else vs
            improved = val < best
            if improved:
                best, best_epoch = val, epoch
                save_checkpoint(ckpt, model, kind, model_kwargs, n, grid, epoch, vd, cfg, vs)
            history.append({
                "epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
                "val_delay": vd, "val_makespan": vs, "val_failures": vfails,
                "train_pp_failures": n_fail, "zero_grad_scenarios": n_zero,
                "seconds": time.time() - t0,
            })
            _log_epoch(tb, model, kind, history[-1], refs)
            log(f"  epoch {epoch:3d}  loss {_fmt(history[-1]['train_loss'])}  val delay {vd:7.2f}  "
                f"makespan {vs:6.2f}"
                f"{'  *best' if improved else '       '}  "
                f"[PP fails val {vfails}/train {n_fail}, zero-grad {n_zero}/{len(idx)}, "
                f"{time.time() - t0:.0f}s]")
    finally:
        runner.close()

    curves = plot_training_curves(history, batch_losses, refs, best_epoch, kind, n, reduction,
                                  os.path.join(tdir, "curves.png"))
    _write_history(history, tdir)
    tb.image_file("curves", curves)
    tb.close()

    final = history[-1]
    log(f"  done: final loss {_fmt(final['train_loss'])}, best val {metric} {best:.2f} at epoch "
        f"{best_epoch} (original PP {refs['orig_pp']:.2f}, {ename} {refs['expert']:.2f})")
    log(f"  model:       {ckpt}")
    log(f"  loss graph:  {curves}")
    log(f"  tensorboard: tensorboard --logdir {os.path.dirname(tensorboard_dir(cfg, grid.name, n, kind))}")
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
    tb.scalar("val/pp_failures", h.get("val_failures"), e)
    tb.scalar("train/pp_failures", h.get("train_pp_failures"), e)
    tb.scalar("train/zero_grad_scenarios", h.get("zero_grad_scenarios"), e)
    tb.scalar("time/epoch_seconds", h.get("seconds"), e)
    if kind == "edge":
        tb.histogram("weights/edge_costs", model.edge_weights.detach().cpu().numpy(), e)


def _write_history(history, out_dir):
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    keys = ["epoch", "train_loss", "val_delay", "val_makespan", "val_failures", "train_pp_failures",
            "zero_grad_scenarios", "seconds"]
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(history)


def save_checkpoint(path, model, kind, model_kwargs, n, grid, epoch, val_delay, cfg,
                    val_makespan=None):
    torch.save({
        "kind": kind, "model_kwargs": model_kwargs, "state_dict": model.state_dict(),
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
