"""Testing on held-out scenarios (paper Table 1) with comparison plots.

For the first ``k`` test scenarios this runs original PP, PP with every trained
model (Edge / GNN) and reads the EECBS expert plan stored with the scenario,
then writes into ``outputs/<map>/n<N>/test/`` (the folder is emptied first, so
every test run fully replaces the previous one):

    summary.md / summary.csv      Table-1 style: mean delay and gap vs EECBS, time, failures
    per_instance.csv              delay of every method on every instance
    summary.png                   bar chart of delays per instance (+ mean)
    instance_000/comparison.png   side-by-side maps: PP (orig) | PP (Edge) | PP (GNN) | EECBS
    instance_000/*.png            each method on its own (and animation.gif if asked)

The same comparison maps and the summary are also logged to TensorBoard
(``tensorboard/test``).

    gap = (mean delay_method - mean delay_EECBS) / mean delay_EECBS * 100
"""

from __future__ import annotations

import csv
import os

import numpy as np
from tqdm import tqdm

from mapf_nco.compare import LABELS, REF, load_models, plan_all_timed
from mapf_nco.grid import Grid
from mapf_nco.metrics import optimality_gap, sum_of_costs
from mapf_nco.paths import fresh_dir, tensorboard_dir, test_dir
from mapf_nco.scenarios import dataset_path, load_dataset
from mapf_nco.solvers.pp import PrioritizedPlanner

METHODS = ("orig", "edge", "gnn")
KEY = {REF: "orig", "PP (Edge)": "edge", "PP (GNN)": "gnn"}


def run_test(cfg, n, num_instances=None, kinds=("edge", "gnn"), plots=True, animate=False,
             log=print):
    from mapf_nco.tb import TB
    from mapf_nco.viz import animate_comparison, plot_comparison, plot_paths, plot_test_summary

    grid = Grid.from_file(cfg["map_path"])
    planner = PrioritizedPlanner(grid, w_min=cfg["w_min"])
    test, info = load_dataset(dataset_path(cfg["data_dir"], grid.name, n, "test"), grid)
    ename = "EECBS" if info.get("expert", "eecbs") == "eecbs" else "CBS"
    k = len(test) if not num_instances else min(int(num_instances), len(test))
    test = test[:k]
    models = load_models(cfg, grid, n, kinds, log)
    out = fresh_dir(test_dir(cfg, grid.name, n))
    tb = TB(tensorboard_dir(cfg, grid.name, n, "test"), log) if plots else None
    log(f"[test] n={n}: {k} test instance(s), methods: "
        + ", ".join([REF] + [LABELS[m] for m in models] + ["Expert"]))

    methods = [REF] + [LABELS[m] for m in models] + ["Expert"]
    per_instance, times = [], {m: [] for m in methods}
    fails = {m: 0 for m in methods}
    say = tqdm.write if log is print else log  # print without breaking the progress bar
    for i, sc in enumerate(tqdm(test, desc=f"test n={n}", unit="inst")):
        plans, t = plan_all_timed(planner, sc, models)
        lb = sc.lower_bound(grid)
        row = {"instance": i, "lower_bound": lb}
        for m in methods:
            paths = plans.get(m)
            if paths is None and m != "Expert":
                fails[m] += 1  # learned weights made PP fail: fall back to original PP
                paths = plans[REF]
            row[m] = None if paths is None else sum_of_costs(paths) - lb
            if m in t:
                times[m].append(t[m])
        per_instance.append(row)
        say(f"  instance {i:3d}: " + "  ".join(
            f"{m} {'-' if row[m] is None else row[m]}" for m in methods))

        if plots:
            idir = os.path.join(out, f"instance_{i:03d}")
            ok = {m: p for m, p in plans.items() if p is not None}
            titles = {m: f"{ename if m == 'Expert' else m}: $\\sum_e y_e$ = {sum_of_costs(p)},  "
                         f"delay {sum_of_costs(p) - lb}"
                      for m, p in ok.items()}
            comp = plot_comparison(grid, sc, ok, titles, os.path.join(idir, "comparison.png"), REF)
            for m, p in ok.items():
                fname = (ename.lower() if m == "Expert" else KEY[m]) + ".png"
                plot_paths(grid, p, sc.starts, sc.goals, titles[m], os.path.join(idir, fname))
            if animate:
                animate_comparison(grid, sc, ok, titles, os.path.join(idir, "animation.gif"))
            tb.image_file(f"comparison/instance_{i:03d}", comp)

    summary = _aggregate(n, per_instance, methods, times, fails, models)
    summary["expert_name"] = ename
    table = format_table([summary])
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n\n" + _per_instance_md(per_instance, methods, ename) + "\n")
    _write_csv(os.path.join(out, "summary.csv"), [summary])
    _write_csv(os.path.join(out, "per_instance.csv"), per_instance)
    chart = plot_test_summary(per_instance, methods, os.path.join(out, "summary.png"),
                              f"Delay on {k} test instances, {n} agents (lower is better)",
                              expert_name=ename)
    if tb is not None:
        tb.image_file("summary", chart)
        tb.text("summary", table)
        tb.close()
    log("\n" + table)
    log(f"\nResults (folder replaced on every test): {out}")
    log("  summary.png, summary.md, per_instance.csv" + (", instance_XXX/comparison.png" if plots else ""))
    return summary


def _aggregate(n, per_instance, methods, times, fails, models):
    expert = [r["Expert"] for r in per_instance if r["Expert"] is not None]
    e = float(np.mean(expert)) if expert else float("nan")
    row = {"agents": n, "num_test": len(per_instance), "expert_delay": e}
    for m in methods:
        if m == "Expert":
            continue
        key = KEY[m]
        d = float(np.mean([r[m] for r in per_instance]))
        row[f"{key}_delay"] = d
        row[f"{key}_gap"] = optimality_gap(d, e) if expert else float("nan")
        row[f"{key}_time"] = float(np.mean(times[m])) if times[m] else float("nan")
        if key != "orig":
            row[f"{key}_failures"] = fails[m]
    return row


def evaluate(cfg, agent_counts, models=("edge", "gnn"), log=print):
    """Table-1 numbers for several agent counts (all test instances, no plots)."""
    return [run_test(cfg, n, None, models, plots=False, log=log) for n in agent_counts]


def format_table(rows):
    """Markdown table in the layout of the paper's Table 1."""
    ename = rows[0].get("expert_name", "EECBS") if rows else "EECBS"
    lines = [
        f"| Agents | PP (orig) | PP (Edge) | PP (GNN) | {ename} |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        cells = []
        for m in METHODS:
            if f"{m}_delay" in r:
                cells.append(f"{r[f'{m}_delay']:.2f} ({r[f'{m}_gap']:.1f}%)")
            else:
                cells.append("—")
        lines.append(f"| {r['agents']} | " + " | ".join(cells) + f" | {r['expert_delay']:.2f} |")
    lines.append("")
    lines.append(f"Average delay (% optimality gap relative to {ename}) over "
                 f"{', '.join(str(r['num_test']) for r in rows)} test instance(s). Lower is better.")
    timing = []
    for r in rows:
        parts = [f"{m}: {r[f'{m}_time'] * 1000:.1f} ms" for m in METHODS if f"{m}_time" in r]
        failures = [f"{m} PP failures: {r[f'{m}_failures']}" for m in ("edge", "gnn")
                    if r.get(f"{m}_failures")]
        timing.append(f"- n={r['agents']}: " + ", ".join(parts + failures))
    return "\n".join(lines + ["", "Mean time per instance (model inference + PP):"] + timing)


def _per_instance_md(per_instance, methods, ename="EECBS"):
    names = [ename if m == "Expert" else m for m in methods]
    lines = ["| instance | " + " | ".join(names) + " |", "|---:|" + "---:|" * len(methods)]
    for r in per_instance:
        lines.append(f"| {r['instance']} | " + " | ".join(
            "—" if r[m] is None else str(r[m]) for m in methods) + " |")
    return "Delay per instance:\n\n" + "\n".join(lines)


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_results(rows, out_dir):
    """Combined Table 1 over several agent counts (used by ``main.py evaluate``)."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "table1.csv")
    _write_csv(csv_path, rows)
    md_path = os.path.join(out_dir, "table1.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(format_table(rows) + "\n")
    return csv_path, md_path
