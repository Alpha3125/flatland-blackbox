"""CBS collision framework: learn edge costs so that CBS resolves fewer conflicts.

CBS's run time grows with the number of conflicts it has to resolve (one per
high-level node it expands). This mode learns edge costs w (the Edge model or the
Cell heat map) for weighted CBS (mapf_nco.solver_cbs.WeightedCBS, which finds the
plan with the smallest total cost sum_e w_e * y_e) so that it resolves fewer
conflicts, and measures what that does to its planning time and to the plan's
real cost (sum of costs on the original graph, where every move and wait costs 1).

Training (black-box differentiation, as in the paper):

    N(w)        number of conflicts CBS resolves with costs w     (the loss we report)
    c           conflict heat: for every conflict CBS resolved, +1 on the edge each
                of the two colliding agents used (mapf_nco.solver_cbs.conflict_edges)
    L(y)        = c . y  (how much the plan uses the edges where collisions happened),
                so dL/dy = c
    w'          = w + lambda * c                                     (perturbed costs)
    dL/dw      ~= -(y(w) - y(w')) / lambda                           (Pogancic et al.)

N itself has no gradient; the surrogate raises the cost of the edges where CBS had to
resolve collisions and moves w toward the plan that avoids them. Two CBS solves per
training scenario. A scenario with no conflict gives no gradient.

Every epoch is checked on the validation scenarios with the current costs: conflicts
resolved, CBS wall-clock time, and the plan's sum of costs on the original graph.
Epoch 0 is unit costs, i.e. plain optimal CBS. The model kept is the epoch with the
fewest conflicts on validation (ties: the faster one).

Sweep: like option 5 (maps, agent scaling, start/goal positions, a grid of settings),
written to outputs/cbs_collision/exp_<k>/:

    cbs_collision_summary.md / .csv / .png       one row per experiment and map
    exp_01_<settings>/<map>/n<N>/models/model_<model>.pt
                               training/training_curves.png, training_history.csv/.json
                               test/test_summary.md, test_per_instance.csv, test_summary.png
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from mapf_nco.grid import Grid
from mapf_nco.metrics import edge_usage
from mapf_nco.solver_cbs import WeightedCBS
from mapf_nco.solver_pp import NoSolutionError

MODELS = ("edge", "cell")
SWEEPABLE = ("model", "epochs", "batch_size", "lr", "lam", "w_min", "val_fraction", "seed",
             "cbs_time_limit")
LABELS = {"edge": "Edge", "cell": "Cell heat map"}

# ----------------------------------------------------------- worker processes
_W = {}


def _init_worker(map_path, w_min, time_limit):
    _W["cbs"] = WeightedCBS(Grid.from_file(map_path), w_min=w_min, time_limit=time_limit)


def _solve(starts, goals, w):
    """Weighted CBS with costs ``w`` (None = unit costs): (paths or None, conflicts
    resolved, seconds, conflict edges)."""
    cbs = _W["cbs"]
    t0 = time.perf_counter()
    try:
        paths = cbs.solve(starts, goals, None, w)
    except NoSolutionError:
        paths = None
    return paths, cbs.last_conflicts, time.perf_counter() - t0, cbs.last_conflict_edges


def _eval_task(args):
    """One validation / test solve: dict(ok, conflicts, seconds, soc, makespan)."""
    starts, goals, w = args
    paths, n, sec, _ = _solve(starts, goals, w)
    if paths is None:
        return {"ok": False, "conflicts": n, "seconds": sec, "soc": -1, "makespan": -1}
    return {"ok": True, "conflicts": n, "seconds": sec,
            "soc": sum(len(p) - 1 for p in paths), "makespan": max(len(p) - 1 for p in paths)}


def _step_task(args):
    """Forward solve, conflict heat, perturbed solve, gradient for one scenario."""
    starts, goals, w, lam = args
    grid = _W["cbs"].grid
    w = np.asarray(w, dtype=np.float64)
    zero = np.zeros_like(w)
    paths, n, sec, hits = _solve(starts, goals, w)
    if paths is None:
        return {"ok": False, "conflicts": n, "seconds": sec, "grad": zero, "pert_ok": False}
    if n == 0:
        return {"ok": True, "conflicts": 0, "seconds": sec, "grad": zero, "pert_ok": True}
    heat = np.bincount(np.asarray(hits, dtype=np.int64), minlength=grid.num_edges)
    y = edge_usage(grid, paths)
    p_paths, _, _, _ = _solve(starts, goals, np.maximum(w + lam * heat, _W["cbs"].w_min))
    if p_paths is None:
        return {"ok": True, "conflicts": n, "seconds": sec, "grad": zero, "pert_ok": False}
    grad = -(y - edge_usage(grid, p_paths)) / lam
    return {"ok": True, "conflicts": n, "seconds": sec, "grad": grad, "pert_ok": True}


class Runner:
    def __init__(self, map_path, w_min, time_limit, workers):
        _init_worker(map_path, w_min, time_limit)
        self.pool = (ProcessPoolExecutor(workers, initializer=_init_worker,
                                         initargs=(map_path, w_min, time_limit))
                     if workers > 1 else None)

    def map(self, fn, tasks):
        return list(map(fn, tasks) if self.pool is None else self.pool.map(fn, tasks))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


# ------------------------------------------------------------------- training
def _summary(results, scenarios, grid):
    """Means over scenarios; a timed-out solve counts with the conflicts it resolved
    so far and its time (the time limit), and is left out of the plan cost."""
    ok = [r for r in results if r["ok"]]
    lbs = [s.lower_bound(grid) for s, r in zip(scenarios, results) if r["ok"]]
    return {
        "conflicts": float(np.mean([r["conflicts"] for r in results])),
        "seconds": float(np.mean([r["seconds"] for r in results])),
        "soc": float(np.mean([r["soc"] for r in ok])) if ok else float("nan"),
        "delay": float(np.mean([r["soc"] - lb for r, lb in zip(ok, lbs)])) if ok
        else float("nan"),
        "makespan": float(np.mean([r["makespan"] for r in ok])) if ok else float("nan"),
        "timeouts": len(results) - len(ok),
    }


def _weights(model, scenarios):
    import torch

    from mapf_nco.models import make_batch

    with torch.no_grad():
        return model(make_batch(scenarios[:1])).detach().cpu().double().numpy()[0]


def train_collision(cfg, n, kind="edge", workers=1, log=print):
    """Train one model for ``n`` agents; returns (checkpoint path, history)."""
    import torch

    from mapf_nco.models import build_model, make_batch
    from mapf_nco.paths import checkpoint_path, training_dir, training_file
    from mapf_nco.scenarios import data_dir_for, dataset_path, load_dataset, train_val_split
    from mapf_nco.train import save_checkpoint, set_seed

    if kind not in MODELS:
        raise ValueError(f"the CBS collision framework learns {' or '.join(MODELS)}, not {kind!r}")
    set_seed(cfg.get("seed", 0))
    grid = Grid.from_file(cfg["map_path"])
    data, _ = load_dataset(dataset_path(data_dir_for(cfg), grid.name, n, "train"), grid)
    train_set, val_set = train_val_split(data, cfg["val_fraction"], cfg.get("seed", 0))
    model = build_model(kind, grid)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lam, B, epochs = float(cfg["lam"]), int(cfg["batch_size"]), int(cfg["epochs"])
    limit = float(cfg.get("cbs_time_limit", 60))
    runner = Runner(cfg["map_path"], cfg["w_min"], limit, workers)
    ckpt = checkpoint_path(cfg, grid.name, n, kind)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    rng = np.random.default_rng(cfg.get("seed", 0))
    log(f"[cbs-collision] {LABELS[kind]}, {n} agents on {grid.name}: {len(train_set)} train / "
        f"{len(val_set)} validation scenarios, {epochs} epochs, batch {B}, lr {cfg['lr']}, "
        f"lambda {lam}, CBS time limit {limit:g} s")

    def validate():
        w = _weights(model, val_set)
        return _summary(runner.map(_eval_task, [(s.starts, s.goals, w) for s in val_set]),
                        val_set, grid)

    history = []
    try:
        v = validate()
        history.append({"epoch": 0, "train_conflicts": None, "train_seconds": None,
                        "zero_grad": None, "train_timeouts": None, "seconds_epoch": None,
                        **{f"val_{k}": x for k, x in v.items()}})
        log(f"  epoch  0  val: conflicts {v['conflicts']:7.1f}  CBS {1000 * v['seconds']:8.1f} ms"
            f"  delay {v['delay']:6.2f}  (unit costs: optimal CBS)")
        best = (v["conflicts"], v["seconds"], 0)
        save_checkpoint(ckpt, model, kind, {}, n, grid, 0, v["delay"], cfg, planner="cbs")
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            idx = rng.permutation(len(train_set))
            confl, secs, n_zero, n_fail = [], [], 0, 0
            for k in range(0, len(idx), B):
                batch = [train_set[i] for i in idx[k:k + B]]
                opt.zero_grad()
                w = model(make_batch(batch))
                w_np = w.detach().cpu().double().numpy()
                res = runner.map(_step_task, [(s.starts, s.goals, w_np[j], lam)
                                              for j, s in enumerate(batch)])
                grads = np.stack([r["grad"] for r in res]) / len(res)
                w.backward(torch.from_numpy(grads).to(w.dtype))
                opt.step()
                confl += [r["conflicts"] for r in res]
                secs += [r["seconds"] for r in res]
                n_zero += sum(not np.any(r["grad"]) for r in res)
                n_fail += sum(not r["ok"] for r in res)
            v = validate()
            h = {"epoch": epoch, "train_conflicts": float(np.mean(confl)),
                 "train_seconds": float(np.mean(secs)), "zero_grad": n_zero,
                 "train_timeouts": n_fail, "seconds_epoch": time.time() - t0,
                 **{f"val_{k}": x for k, x in v.items()}}
            history.append(h)
            mark = ""
            if (v["conflicts"], v["seconds"]) < best[:2]:
                best = (v["conflicts"], v["seconds"], epoch)
                save_checkpoint(ckpt, model, kind, {}, n, grid, epoch, v["delay"], cfg,
                                planner="cbs")
                mark = "  *best"
            log(f"  epoch {epoch:2d}  train conflicts {h['train_conflicts']:7.1f}  val: conflicts "
                f"{v['conflicts']:7.1f}  CBS {1000 * v['seconds']:8.1f} ms  delay "
                f"{v['delay']:6.2f}  [timeouts val {v['timeouts']}/train {n_fail}, zero-grad "
                f"{n_zero}/{len(train_set)}, {h['seconds_epoch']:.0f}s]{mark}")
    finally:
        runner.close()
    tdir = training_dir(cfg, grid.name, n)
    os.makedirs(tdir, exist_ok=True)
    with open(training_file(cfg, grid.name, n, "history.json"), "w") as f:
        json.dump({"model": kind, "best_epoch": best[2], "history": history}, f, indent=1)
    with open(training_file(cfg, grid.name, n, "history.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(history[0]))
        wr.writeheader()
        wr.writerows(history)
    plot_training(history, best[2], kind, grid.name, n, cfg,
                  training_file(cfg, grid.name, n, "curves.png"))
    log(f"  done: fewest validation conflicts at epoch {best[2]} ({best[0]:.1f}, against "
        f"{history[0]['val_conflicts']:.1f} with unit costs); model {ckpt}")
    return ckpt, {"best_epoch": best[2], "history": history}


# --------------------------------------------------------------------- testing
def test_collision(cfg, n, kind, num=None, workers=1, log=print):
    """Original CBS (unit costs) vs CBS on the learned costs, on the saved test scenarios."""
    from mapf_nco.paths import checkpoint_path, fresh_dir, test_dir, test_file
    from mapf_nco.scenarios import data_dir_for, dataset_path, load_dataset
    from mapf_nco.train import load_model

    grid = Grid.from_file(cfg["map_path"])
    test, _ = load_dataset(dataset_path(data_dir_for(cfg), grid.name, n, "test"), grid)
    test = test[:num] if num else test
    model, ck = load_model(checkpoint_path(cfg, grid.name, n, kind), grid)
    w = _weights(model, test)
    limit = float(cfg.get("cbs_time_limit", 60))
    runner = Runner(cfg["map_path"], cfg["w_min"], limit, workers)
    try:
        # unit costs through the same weighted code path as the learned costs (as epoch 0),
        # so tie-breaking and timing differences come from the costs only
        ones = np.ones(grid.num_edges)
        orig = runner.map(_eval_task, [(s.starts, s.goals, ones) for s in test])
        learned = runner.map(_eval_task, [(s.starts, s.goals, w) for s in test])
    finally:
        runner.close()
    so, sl = _summary(orig, test, grid), _summary(learned, test, grid)
    out = test_dir(cfg, grid.name, n)
    fresh_dir(out)
    rows = []
    for i, (s, a, b) in enumerate(zip(test, orig, learned)):
        lb = s.lower_bound(grid)
        rows.append({"instance": i, "lower_bound": lb,
                     "orig_conflicts": a["conflicts"], "learned_conflicts": b["conflicts"],
                     "orig_ms": round(1000 * a["seconds"], 1),
                     "learned_ms": round(1000 * b["seconds"], 1),
                     "orig_delay": a["soc"] - lb if a["ok"] else "timeout",
                     "learned_delay": b["soc"] - lb if b["ok"] else "timeout"})
    with open(test_file(out, "per_instance.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    md = ["| | conflicts resolved | CBS time (ms) | delay (SoC - lower bound) | makespan | "
          "timeouts |", "|---|---|---|---|---|---|"]
    for name, s in (("CBS, unit costs (optimal)", so), (f"CBS, learned costs ({LABELS[kind]}, "
                                                       f"epoch {ck.get('epoch')})", sl)):
        md.append(f"| {name} | {s['conflicts']:.1f} | {1000 * s['seconds']:.1f} | "
                  f"{s['delay']:.2f} | {s['makespan']:.2f} | {s['timeouts']} |")
    with open(test_file(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# CBS collision framework: {grid.name}, {n} agents, {len(test)} test "
                f"scenarios\n\nMeans per scenario. The delay is measured on the original graph "
                f"(every move and wait costs 1). A timed-out solve counts with its time limit "
                f"and the conflicts resolved so far, and is left out of delay and makespan."
                f"\n\n" + "\n".join(md) + "\n")
    plot_test(rows, kind, grid.name, n, test_file(out, "summary.png"))
    log(f"[test] {len(test)} scenarios: conflicts {so['conflicts']:.1f} -> {sl['conflicts']:.1f}, "
        f"CBS {1000 * so['seconds']:.1f} -> {1000 * sl['seconds']:.1f} ms, delay "
        f"{so['delay']:.2f} -> {sl['delay']:.2f} (timeouts {so['timeouts']} -> {sl['timeouts']})")
    return {"num_test": len(test), "best_epoch": ck.get("epoch"),
            **{f"orig_{k}": x for k, x in so.items()}, **{f"learned_{k}": x for k, x in sl.items()}}


# ----------------------------------------------------------------------- sweep
def run_collision_sweep(cfg, n, grid_values, test_count=None, workers=None, log=print,
                        maps=None, scalings=None):
    """Sweep for the CBS collision framework (see the module docstring)."""
    from mapf_nco.experts import generate_datasets
    from mapf_nco.scenarios import positions_of
    from mapf_nco.sweep import _short, _val, expand_grid, next_sweep_dir, scp_command, sweep_targets

    bad = [m for m in grid_values.get("model", [cfg.get("model", "edge")]) if m not in MODELS]
    if bad:
        raise ValueError(f"the CBS collision framework learns {' or '.join(MODELS)}, not {bad}")
    targets = sweep_targets(cfg, n, maps, scalings)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    positions = positions_of(cfg)
    if positions == "fixed":
        test_count = 1  # every test instance is the same scenario
    experiments = expand_grid(grid_values)
    varying = [k for k, v in grid_values.items() if len(v) > 1]
    root = next_sweep_dir(os.path.join(cfg["output_dir"], "cbs_collision"))
    os.makedirs(root)
    log(f"[cbs-collision] {len(experiments)} experiment(s) on "
        + "; ".join(f"{t['grid'].name} with {t['agents']} agents" for t in targets)
        + f"; start/goal positions: {positions}. Results: {root}")
    for t in targets:
        log(f"\n[cbs-collision] data: {t['grid'].name}, {t['agents']} agents")
        generate_datasets(dict(cfg, map_path=t["map_path"], augment=True), [t["agents"]],
                          workers=workers, log=log)
    rows = []
    for i, settings in enumerate(experiments, 1):
        name = f"exp_{i:02d}"
        folder = name + "".join(f"_{_short(k)}-{_val(settings[k])}" for k in varying)
        for t in targets:
            log(f"\n{'=' * 70}\n[cbs-collision] {name}/{len(experiments):02d} on {t['grid'].name}, "
                f"{t['agents']} agents: " + ", ".join(f"{k}={v}" for k, v in settings.items())
                + f"\n{'=' * 70}")
            ecfg = dict(cfg, map_path=t["map_path"], **settings)
            ecfg["output_dir"] = os.path.join(root, folder)
            kind = ecfg.get("model", "edge")
            t0 = time.time()
            _, hist = train_collision(ecfg, t["agents"], kind, workers, log)
            s = test_collision(ecfg, t["agents"], kind, test_count, workers, log)
            h0 = hist["history"][0]
            hb = hist["history"][hist["best_epoch"]]
            rows.append({
                "experiment": name, "map": t["grid"].name, "agents": t["agents"],
                "positions": positions, **settings, "model": kind,
                "best_epoch": hist["best_epoch"],
                "val_conflicts_start": round(h0["val_conflicts"], 2),
                "val_conflicts_best": round(hb["val_conflicts"], 2),
                "test_scenarios": s["num_test"],
                **{f"test_{k}": round(v, 4) if isinstance(v, float) else v
                   for k, v in s.items() if k.startswith(("orig_", "learned_"))},
                "minutes": round((time.time() - t0) / 60, 1)})
            _write_summary(root, rows)
    plot_summary(rows, os.path.join(root, "cbs_collision_summary.png"))
    log(f"\n[cbs-collision] done. {os.path.join(root, 'cbs_collision_summary.md')}")
    with open(os.path.join(root, "cbs_collision_summary.md"), encoding="utf-8") as f:
        log(f.read())
    cmd = scp_command(cfg, root)
    if cmd:
        log("[cbs-collision] to copy the results to your PC, run this on the PC:\n\n" + cmd + "\n")
    return root, rows


def _write_summary(root, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(root, "cbs_collision_summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    settings = [k for k in keys if k in SWEEPABLE]
    head = (["exp", "map", "agents"] + settings + ["best ep.", "conflicts: unit / learned",
            "CBS ms: unit / learned", "delay: unit (optimal) / learned", "timeouts: unit / learned",
            "min"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["experiment"][4:], r["map"], str(r["agents"])] + [str(r.get(k, "-"))
                                                                    for k in settings]
        cells += [str(r["best_epoch"]),
                  f"{r['test_orig_conflicts']:.1f} / {r['test_learned_conflicts']:.1f}",
                  f"{1000 * r['test_orig_seconds']:.1f} / {1000 * r['test_learned_seconds']:.1f}",
                  f"{r['test_orig_delay']:.2f} / {r['test_learned_delay']:.2f}",
                  f"{r['test_orig_timeouts']} / {r['test_learned_timeouts']}", str(r["minutes"])]
        lines.append("| " + " | ".join(cells) + " |")
    pos = rows[0].get("positions", "randomized")
    with open(os.path.join(root, "cbs_collision_summary.md"), "w", encoding="utf-8") as f:
        f.write("# CBS collision framework\n\nStart/goal positions: " + pos + ". Test-set means "
                "per scenario: conflicts CBS resolves, CBS wall-clock time, and the plan's delay "
                "on the original graph (sum of costs - lower bound; unit costs give the optimal "
                "plan). Lower is better.\n\n" + "\n".join(lines) + "\n")


# ----------------------------------------------------------------------- plots
_TEXT, _MUTED, _GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
_BLUE, _ORANGE = "#257de0", "#f47230"


def _style(ax):
    ax.grid(color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=_MUTED)


def plot_training(history, best, kind, map_name, n, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [
        ("Conflicts CBS resolves (per scenario)", "val_conflicts", "train_conflicts", 1),
        ("CBS planning time (ms per scenario)", "val_seconds", "train_seconds", 1000),
        ("Plan cost on the original graph: delay", "val_delay", None, 1)]
    for ax, (title, vk, tk, scale) in zip(axes, panels):
        ax.plot(ep, [scale * h[vk] for h in history], color=_BLUE, lw=2, marker="o", ms=5,
                label="validation")
        if tk:
            te = [h["epoch"] for h in history if h[tk] is not None]
            ax.plot(te, [scale * h[tk] for h in history if h[tk] is not None], color=_ORANGE,
                    lw=2, marker="s", ms=5, ls="--", label="training (during the epoch)")
        ax.axhline(scale * history[0][vk], color=_MUTED, ls=":", lw=1.5,
                   label="unit costs (optimal CBS)")
        ax.plot([best], [scale * history[best][vk]], marker="*", ms=16, color=_BLUE,
                markeredgecolor=_TEXT, ls="none", zorder=5, label="kept epoch")
        ax.set_title(title, fontsize=11, color=_TEXT)
        ax.set_xlabel("epoch")
        ax.set_xticks(ep)
        _style(ax)
    axes[2].set_ylabel("sum of costs - lower bound (steps)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"CBS collision framework: {LABELS[kind]}, {map_name}, {n} agents "
                 f"(lr {cfg['lr']}, batch {cfg['batch_size']}, λ {cfg['lam']})", fontsize=12,
                 color=_TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def plot_test(rows, kind, map_name, n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (title, a, b) in zip(axes, [("Conflicts CBS resolves", "orig_conflicts",
                                         "learned_conflicts"),
                                        ("CBS time (ms)", "orig_ms", "learned_ms"),
                                        ("Delay on the original graph", "orig_delay",
                                         "learned_delay")]):
        va = [r[a] if r[a] != "timeout" else np.nan for r in rows]
        vb = [r[b] if r[b] != "timeout" else np.nan for r in rows]
        ax.bar(x - 0.2, va, 0.4, color=_MUTED, label="unit costs (optimal CBS)")
        ax.bar(x + 0.2, vb, 0.4, color=_BLUE, label=f"learned ({LABELS[kind]})")
        ax.set_title(title, fontsize=11, color=_TEXT)
        ax.set_xlabel("test instance")
        if len(rows) <= 30:
            ax.set_xticks(x)
        _style(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"CBS collision framework, test: {map_name}, {n} agents", fontsize=12,
                 color=_TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def plot_summary(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = list(dict.fromkeys((r["map"], r["agents"]) for r in rows))
    fig, axes = plt.subplots(len(targets), 3, figsize=(15, 4.2 * len(targets)), squeeze=False)
    for row_axes, (m, a) in zip(axes, targets):
        rr = [r for r in rows if (r["map"], r["agents"]) == (m, a)]
        x = np.arange(len(rr))
        for ax, (title, key, scale) in zip(row_axes, [("conflicts resolved", "conflicts", 1),
                                                      ("CBS time (ms)", "seconds", 1000),
                                                      ("delay (original graph)", "delay", 1)]):
            ax.bar(x, [scale * r[f"test_learned_{key}"] for r in rr], 0.6, color=_BLUE,
                   label="learned costs")
            ax.axhline(scale * rr[0][f"test_orig_{key}"], color=_MUTED, ls="--", lw=1.5,
                       label="unit costs (optimal CBS)")
            ax.set_xticks(x, [r["experiment"][4:] for r in rr])
            ax.set_title(f"{m}, {a} agents: {title}", fontsize=10, color=_TEXT)
            ax.set_xlabel("experiment")
            _style(ax)
        row_axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
