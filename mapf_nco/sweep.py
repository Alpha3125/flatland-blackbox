"""Sweep: one dataset, a queue of experiments, and the same tests for each.

For every combination of the hyperparameter values given (a grid search), the
chosen model (``model``: edge / potential / potential_edge) is trained from
scratch, for PP, CBS or both (``planner``), and then tested on

  1. the saved test scenarios (same agent count as training), and
  2. optionally, a second set of new random scenarios with a different agent
     count (off by default: every experiment then uses the selected agent count
     only). This set is generated once and shared by all experiments, so they
     are compared fairly.

Everything goes to ``outputs/sweeps/exp_<k>/`` (k = the smallest free number, so deleting
exp_2 makes the next sweep exp_2 again), and each experiment to
``exp_01_<the settings that vary>/`` inside it:

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
import re
import time

from mapf_nco.grid import Grid
from mapf_nco.planners import KIND_LABELS, MODEL_IDS, kind_of, model_ids, planner_of, training_runs
from mapf_nco.scenarios import load_dataset, save_dataset

# Hyperparameters that can be swept.
SWEEPABLE = ("model", "planner", "objective", "makespan_tau", "makespan_weight", "epochs", "batch_size",
             "lr", "lam", "loss_reduction", "w_min", "val_fraction", "seed")


_SHORT = {"loss_reduction": "loss", "makespan_tau": "tau", "makespan_weight": "mw",
          "batch_size": "batch", "val_fraction": "val"}


def _short(key):
    return _SHORT.get(key, key)


def _val(v):
    return str(v).replace("+", "-").replace("/", "-")


def next_sweep_dir(base):
    """``base/exp_<k>`` with k the smallest natural number not used yet (exp_1, exp_2, ...)."""
    used = set()
    if os.path.isdir(base):
        for name in os.listdir(base):
            m = re.fullmatch(r"exp_(\d+)", name)
            if m:
                used.add(int(m.group(1)))
    k = 1
    while k in used:
        k += 1
    return os.path.join(base, f"exp_{k}")


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
    return {**cfg, **settings}


def run_sweep(cfg, n, grid_values, test_count=None, second_n=None, second_count=None,
              second_seed=777, animate=False, workers=None, log=print):
    from mapf_nco.evaluate import run_test, test_saved
    from mapf_nco.experts import expert_options, make_scenarios
    from mapf_nco.train import train
    from mapf_nco.viz import plot_sweep_summary

    grid = Grid.from_file(cfg["map_path"])
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    experiments = expand_grid(grid_values)
    root = next_sweep_dir(os.path.join(cfg["output_dir"], "sweeps"))
    varying = [k for k, v in grid_values.items() if len(v) > 1]
    os.makedirs(root)
    log(f"[sweep] {len(experiments)} experiment(s), {n} agents"
        + (f"; second test on {second_n} agents" if second_n else "") + f". Results: {root}")

    second, ename = None, "EECBS" if cfg.get("expert", "eecbs") == "eecbs" else "CBS"
    if second_n:  # shared second test set (generated once)
        expert, opts = expert_options(cfg, grid, log)
        test_size = second_count or test_count or cfg.get("num_test", 25)
        second_path = os.path.join(root, "second_test.json.gz")
        second, _, _ = make_scenarios(cfg, second_n, test_size, 20_000_000 + second_seed, expert,
                                      opts, workers, desc=f"second test set ({second_n} agents)")
        save_dataset(second_path, grid, second, info={"num_agents": second_n, "expert": expert})
        second, _ = load_dataset(second_path, grid)
    tags = [f"n{n}"] + ([f"n{second_n}"] if second_n else [])

    rows = []
    for i, settings in enumerate(experiments, 1):
        name = f"exp_{i:02d}"
        folder = name + "".join(f"_{_short(k)}-{_val(settings[k])}" for k in varying)
        log(f"\n{'=' * 70}\n[sweep] {name}/{len(experiments):02d}: "
            + ", ".join(f"{k}={v}" for k, v in settings.items()) + f"\n{'=' * 70}")
        ecfg = apply_settings(cfg, settings)
        ecfg["output_dir"] = os.path.join(root, folder)
        row = {"experiment": name, **settings}
        t0 = time.time()
        for kind, planner in training_runs(ecfg):
            train(ecfg, n, planner, workers=workers, log=log, kind=kind)
        ids = model_ids(ecfg)
        s1 = test_saved(ecfg, n, test_count, ids, plots=True, animate=animate, log=log)
        summaries = [s1]
        if second_n:
            summaries.append(run_test(ecfg, grid, n, second, ename,
                                      f"{len(second)} new random scenarios, {second_n} agents",
                                      ids, True, animate, log,
                                      out_name=f"test_n{second_n}"))
        row.update(_best_epochs(ecfg, grid, n, ids))
        for tag, s in zip(tags, summaries):
            for m in ("orig", "cbs") + MODEL_IDS:
                if f"{m}_delay" not in s:
                    continue
                row[f"{tag}_{m}_delay"] = round(s.get(f"{m}_delay", float("nan")), 2)
                row[f"{tag}_{m}_makespan"] = round(s.get(f"{m}_makespan", float("nan")), 2)
                if m not in ("orig", "cbs"):
                    row[f"{tag}_{m}_gap"] = round(s.get(f"{m}_gap", float("nan")), 1)
            row[f"{tag}_expert_delay"] = round(s["expert_delay"], 2)
            row[f"{tag}_expert_makespan"] = round(s.get("expert_makespan", float("nan")), 2)
        row["minutes"] = round((time.time() - t0) / 60, 1)
        rows.append(row)
        _write_summary(root, rows, tags, ename)  # updated after every experiment

    plot_sweep_summary(rows, tags, os.path.join(root, "sweep_summary.png"))
    log(f"\n[sweep] done. {os.path.join(root, 'sweep_summary.md')}")
    with open(os.path.join(root, "sweep_summary.md"), encoding="utf-8") as f:
        log(f.read())
    return root, rows


def _best_epochs(cfg, grid, n, ids):
    import torch

    from mapf_nco.paths import checkpoint_path

    out = {}
    for kind in ids:
        ck = torch.load(checkpoint_path(cfg, grid.name, n, kind), map_location="cpu",
                        weights_only=False)
        out[f"{kind}_best_epoch"] = ck.get("epoch")
        out[f"{kind}_val_delay"] = round(float(ck.get("val_delay", float("nan"))), 2)
    return out


def _write_summary(root, rows, tags, ename):
    if ename == "CBS":  # Python CBS as the expert: don't confuse it with the original-CBS column
        ename = "CBS (expert)"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(root, "sweep_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    settings = [k for k in keys if k in SWEEPABLE]
    ids = [m for m in MODEL_IDS
           if any(f"{m}_best_epoch" in r for r in rows)]
    names = {m: f"{planner_of(m).upper()} {KIND_LABELS[kind_of(m)]}" for m in MODEL_IDS}
    pp = [m for m in ids if planner_of(m) == "pp"]
    cbs = [m for m in ids if planner_of(m) == "cbs"]
    groups = []  # (column title, per-tag keys)
    if pp:
        groups.append(("PP / " + " / ".join(names[m] for m in pp) + f" / {ename}",
                       ["orig"] + pp + ["expert"]))
    if cbs:
        tail = [] if pp else ["expert"]
        groups.append(("CBS / " + " / ".join(names[m] for m in cbs)
                       + ("" if pp else f" / {ename}"), ["cbs"] + cbs + tail))
    if not groups:
        groups.append((f"PP / {ename}", ["orig", "expert"]))
    head = (["exp"] + settings + ["best ep. " + " / ".join(names[m] for m in ids),
                                  "val delay " + " / ".join(names[m] for m in ids)]
            + [f"delay {t}: {g}" for t in tags for g, _ in groups]
            + [f"makespan {t}: {g}" for t in tags for g, _ in groups] + ["min"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["experiment"][4:]] + [str(r[k]) for k in settings]
        cells.append(" / ".join(str(r.get(f"{m}_best_epoch", "-")) for m in ids))
        cells.append(" / ".join(str(r.get(f"{m}_val_delay", "-")) for m in ids))
        for what in ("delay", "makespan"):
            for tag in tags:
                for _, cols in groups:
                    cells.append(" / ".join(str(r.get(f"{tag}_{m}_{what}", "-")) for m in cols))
        cells.append(str(r["minutes"]))
        lines.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(root, "sweep_summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# Sweep: trained on {tags[0][1:]} agents\n\nAverage delay and makespan on the "
                f"test set(s) {', '.join(tags)}. PP columns: original PP / learned PP / expert; "
                f"CBS columns: original CBS / CBS with learned costs. Lower is better.\n\n"
                + "\n".join(lines) + "\n")
