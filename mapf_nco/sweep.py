"""Sweep: one dataset, a queue of experiments, and the same tests for each.

For every combination of the hyperparameter values given (a grid search), the
Edge and GNN models are trained from scratch and then tested on

  1. the saved test scenarios (same agent count as training), and
  2. a second set of new random scenarios with a different agent count:
     half the agents when training used >= 50, otherwise a larger count
     (default: the largest agent count in the config). This set is generated
     once and shared by all experiments, so they are compared fairly.

Everything goes to ``outputs/sweeps/<map>_n<N>_<timestamp>/``:

    sweep_summary.md / .csv          one row per experiment: settings, best epochs,
                                     validation delay, test delays and gaps
    sweep_summary.png                test delay of every experiment (bar chart)
    second_test.json.gz              the shared second test set
    exp_01/ ... exp_K/<map>/n<N>/    checkpoints, training curves, TensorBoard,
                                     test/ and test_n<M>/ comparison plots
"""

from __future__ import annotations

import csv
import itertools
import os
import time

from mapf_nco.grid import Grid
from mapf_nco.scenarios import load_dataset, save_dataset

# Hyperparameters that can be swept; "gnn_*" map to cfg["gnn"][*].
SWEEPABLE = ("objective", "makespan_tau", "makespan_weight", "epochs", "batch_size", "lr",
             "lam", "loss_reduction", "w_min", "val_fraction", "seed", "gnn_hidden", "gnn_num_freqs", "gnn_id_dim", "gnn_mlp_hidden")


def second_agent_count(n, cfg):
    """Half the agents for n >= 50, otherwise the largest configured agent count."""
    if n >= 50:
        return max(1, n // 2)
    return max([int(x) for x in cfg.get("agent_counts", [100])] + [n])


def expand_grid(grid_values):
    """{'lr': [1e-4, 5e-4], 'lam': [1.0]} -> list of dicts (cartesian product)."""
    keys = list(grid_values)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid_values[k] for k in keys))]


def apply_settings(cfg, settings):
    cfg = dict(cfg)
    gnn = dict(cfg.get("gnn", {}))
    for k, v in settings.items():
        if k.startswith("gnn_"):
            gnn[k[4:]] = v
        else:
            cfg[k] = v
    cfg["gnn"] = gnn
    return cfg


def run_sweep(cfg, n, grid_values, test_count=None, second_n=None, second_count=None,
              second_seed=777, animate=False, workers=None, log=print):
    from mapf_nco.evaluate import run_test, test_saved
    from mapf_nco.experts import expert_options, make_scenarios
    from mapf_nco.train import train
    from mapf_nco.viz import plot_sweep_summary

    grid = Grid.from_file(cfg["map_path"])
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    experiments = expand_grid(grid_values)
    second_n = second_n or second_agent_count(n, cfg)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = os.path.join(cfg["output_dir"], "sweeps", f"{grid.name}_n{n}_{stamp}")
    os.makedirs(root, exist_ok=True)
    log(f"[sweep] {len(experiments)} experiment(s), trained on {n} agents; "
        f"second test on {second_n} agents. Results: {root}")

    # Shared second test set (generated once).
    expert, opts = expert_options(cfg, grid, log)
    test_size = second_count or test_count or cfg.get("num_test", 25)
    second_path = os.path.join(root, "second_test.json.gz")
    second, _, _ = make_scenarios(cfg, second_n, test_size, 20_000_000 + second_seed, expert,
                                  opts, workers, desc=f"second test set ({second_n} agents)")
    save_dataset(second_path, grid, second, info={"num_agents": second_n, "expert": expert})
    second, _ = load_dataset(second_path, grid)
    ename = "EECBS" if expert == "eecbs" else "CBS"

    rows = []
    for i, settings in enumerate(experiments, 1):
        name = f"exp_{i:02d}"
        log(f"\n{'=' * 70}\n[sweep] {name}/{len(experiments):02d}: "
            + ", ".join(f"{k}={v}" for k, v in settings.items()) + f"\n{'=' * 70}")
        ecfg = apply_settings(cfg, settings)
        ecfg["output_dir"] = os.path.join(root, name)
        row = {"experiment": name, **settings}
        t0 = time.time()
        for kind in ("edge", "gnn"):
            train(ecfg, n, kind, workers=workers, log=log)
        s1 = test_saved(ecfg, n, test_count, ("edge", "gnn"), plots=True, animate=animate,
                        log=log)
        s2 = run_test(ecfg, grid, n, second, ename,
                      f"{len(second)} new random scenarios, {second_n} agents",
                      ("edge", "gnn"), True, animate, log, out_name=f"test_n{second_n}")
        row.update(_best_epochs(ecfg, grid, n))
        for tag, s in ((f"n{n}", s1), (f"n{second_n}", s2)):
            for m in ("orig", "edge", "gnn"):
                row[f"{tag}_{m}_delay"] = round(s.get(f"{m}_delay", float("nan")), 2)
                row[f"{tag}_{m}_makespan"] = round(s.get(f"{m}_makespan", float("nan")), 2)
                if m != "orig":
                    row[f"{tag}_{m}_gap"] = round(s.get(f"{m}_gap", float("nan")), 1)
            row[f"{tag}_expert_delay"] = round(s["expert_delay"], 2)
            row[f"{tag}_expert_makespan"] = round(s.get("expert_makespan", float("nan")), 2)
        row["minutes"] = round((time.time() - t0) / 60, 1)
        rows.append(row)
        _write_summary(root, rows, n, second_n, ename)  # updated after every experiment

    plot_sweep_summary(rows, n, second_n, os.path.join(root, "sweep_summary.png"))
    log(f"\n[sweep] done. {os.path.join(root, 'sweep_summary.md')}")
    with open(os.path.join(root, "sweep_summary.md"), encoding="utf-8") as f:
        log(f.read())
    return root, rows


def _best_epochs(cfg, grid, n):
    import torch

    from mapf_nco.paths import checkpoint_path

    out = {}
    for kind in ("edge", "gnn"):
        ck = torch.load(checkpoint_path(cfg, grid.name, n, kind), map_location="cpu",
                        weights_only=False)
        out[f"{kind}_best_epoch"] = ck.get("epoch")
        out[f"{kind}_val_delay"] = round(float(ck.get("val_delay", float("nan"))), 2)
    return out


def _write_summary(root, rows, n, second_n, ename):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(root, "sweep_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    settings = [k for k in keys if k in SWEEPABLE]
    head = (["exp"] + settings + ["best ep. Edge/GNN", "val delay Edge/GNN",
                                  f"test n={n}: PP / Edge / GNN / {ename}",
                                  f"test n={second_n}: PP / Edge / GNN / {ename}",
                                  f"makespan n={n}", f"makespan n={second_n}", "min"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["experiment"][4:]] + [str(r[k]) for k in settings]
        cells.append(f"{r['edge_best_epoch']} / {r['gnn_best_epoch']}")
        cells.append(f"{r['edge_val_delay']} / {r['gnn_val_delay']}")
        for tag in (f"n{n}", f"n{second_n}"):
            cells.append(f"{r[f'{tag}_orig_delay']} / {r[f'{tag}_edge_delay']} / "
                         f"{r[f'{tag}_gnn_delay']} / {r[f'{tag}_expert_delay']}")
        for tag in (f"n{n}", f"n{second_n}"):
            cells.append(f"{r[f'{tag}_orig_makespan']} / {r[f'{tag}_edge_makespan']} / "
                         f"{r[f'{tag}_gnn_makespan']} / {r[f'{tag}_expert_makespan']}")
        cells.append(str(r["minutes"]))
        lines.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(root, "sweep_summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# Sweep: trained on {n} agents\n\nAverage delay and makespan, each as "
                f"PP / Edge / GNN / expert (lower is better). Second test "
                f"set: {second_n} agents, shared by all experiments.\n\n" + "\n".join(lines) + "\n")
