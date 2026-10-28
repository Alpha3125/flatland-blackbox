"""Testing (paper Table 1) with comparison plots.

Two ways to test the trained models:

* ``test_saved``: the generated test scenarios (same agent count the models
  were trained on), with their pre-computed EECBS / CBS / PP plans;
* ``test_random``: brand-new random scenarios with any number of agents,
  solved by EECBS (and CBS) on the spot. This checks that the models generalise
  to start/goal positions and agent counts they have never seen.

Every test empties and rewrites ``outputs/<map>/n<N>/test/``:

    summary.md / summary.csv      Table-1 style: mean delay, gap vs EECBS, time, failures
    per_instance.csv              delay of every method on every instance
    summary.png                   bar chart of delays per instance (+ mean)
    instance_000/comparison.png   top: PP (orig) | PP (Edge) | PP (GNN) | EECBS | CBS
                                  bottom: the edge costs each model assigns (gradient)
                                  header: the training hyperparameters
    instance_000/*.png            each method on its own (and animation.gif if asked)

The comparison maps and the summary are also logged to TensorBoard (``tensorboard/test``).

    gap = (mean delay_method - mean delay_EECBS) / mean delay_EECBS * 100
"""

from __future__ import annotations

import csv
import os

import numpy as np
from tqdm import tqdm

from mapf_nco.compare import LABELS, REF, WEIGHT_LABELS, hparams_header, load_models, \
    plan_all_timed
from mapf_nco.grid import Grid
from mapf_nco.metrics import optimality_gap, sum_of_costs
from mapf_nco.paths import fresh_dir, tensorboard_dir, test_dir
from mapf_nco.scenarios import dataset_path, load_dataset
from mapf_nco.solver_pp import PrioritizedPlanner

METHODS = ("orig", "edge", "gnn", "cbs")
KEY = {REF: "orig", "PP (Edge)": "edge", "PP (GNN)": "gnn", "CBS": "cbs"}


def test_saved(cfg, n, num_instances=None, kinds=("edge", "gnn"), plots=True, animate=False,
               log=print):
    """Test the models trained on ``n`` agents on the generated test scenarios."""
    grid = Grid.from_file(cfg["map_path"])
    test, info = load_dataset(dataset_path(cfg["data_dir"], grid.name, n, "test"), grid)
    k = len(test) if not num_instances else min(int(num_instances), len(test))
    ename = "EECBS" if info.get("expert", "eecbs") == "eecbs" else "CBS"
    return run_test(cfg, grid, n, test[:k], ename, f"{k} saved test scenarios, {n} agents",
                    kinds, plots, animate, log)


def test_random(cfg, trained_n, num_agents, count, seed=12345, kinds=("edge", "gnn"),
                plots=True, animate=False, workers=None, log=print):
    """Test the models trained on ``trained_n`` agents on new random scenarios."""
    from mapf_nco.experts import expert_options, make_scenarios, suboptimality_for

    grid = Grid.from_file(cfg["map_path"])
    expert, opts = expert_options(cfg, grid, log)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    w = suboptimality_for(num_agents, cfg)
    log(f"[test] solving {count} new random scenarios with {num_agents} agents "
        f"({'EECBS w=' + str(w) if expert == 'eecbs' else 'Python CBS'}, plus CBS)")
    scenarios, _, _ = make_scenarios(cfg, num_agents, count, 10_000_000 + seed, expert, opts,
                                     workers, desc="new scenarios")
    ename = "EECBS" if expert == "eecbs" else "CBS"
    return run_test(cfg, grid, trained_n, scenarios, ename,
                    f"{count} new random scenarios, {num_agents} agents (seed {seed})",
                    kinds, plots, animate, log)


def run_test(cfg, grid, trained_n, scenarios, ename, label, kinds=("edge", "gnn"),
             plots=True, animate=False, log=print):
    from mapf_nco.tb import TB
    from mapf_nco.viz import animate_comparison, plot_comparison, plot_paths, plot_test_summary

    planner = PrioritizedPlanner(grid, w_min=cfg["w_min"])
    models = load_models(cfg, grid, trained_n, kinds, log)
    header = hparams_header(models, trained_n)
    out = fresh_dir(test_dir(cfg, grid.name, trained_n))
    tb = TB(tensorboard_dir(cfg, grid.name, trained_n, "test"), log) if plots else None
    has_cbs = any(s.cbs_paths for s in scenarios)
    methods = [REF] + [LABELS[m] for m in models] + ["Expert"] + (["CBS"] if has_cbs else [])
    names = {m: (ename if m == "Expert" else m) for m in methods}
    log(f"[test] {label}; methods: " + ", ".join(names.values()))
    log(f"       {header}")

    per_instance, times = [], {m: [] for m in methods}
    fails = {m: 0 for m in methods}
    say = tqdm.write if log is print else log  # print without breaking the progress bar
    for i, sc in enumerate(tqdm(scenarios, desc="test", unit="inst")):
        plans, t, weights = plan_all_timed(planner, sc, models)
        lb = sc.lower_bound(grid)
        row = {"instance": i, "agents": sc.num_agents, "lower_bound": lb}
        for m in methods:
            paths = plans.get(m)
            if paths is None and m in KEY and m != "CBS":
                fails[m] += 1  # learned weights made PP fail: fall back to original PP
                paths = plans[REF]
            row[m] = None if paths is None else sum_of_costs(paths) - lb
            if m in t:
                times[m].append(t[m])
        per_instance.append(row)
        say(f"  instance {i:3d}: " + "  ".join(
            f"{names[m]} {'-' if row[m] is None else row[m]}" for m in methods))

        if plots:
            idir = os.path.join(out, f"instance_{i:03d}")
            ok = {m: p for m, p in plans.items() if p is not None}
            titles = {m: f"{names.get(m, m)}: $\\sum_e y_e$ = {sum_of_costs(p)},  "
                         f"delay {sum_of_costs(p) - lb}" for m, p in ok.items()}
            comp = plot_comparison(
                grid, sc, ok, titles, os.path.join(idir, "comparison.png"), REF,
                weights=weights, weight_titles=WEIGHT_LABELS,
                header=f"{label}, instance {i}  ||  {header}")
            for m, p in ok.items():
                fname = (ename.lower() if m == "Expert" else KEY[m]) + ".png"
                plot_paths(grid, p, sc.starts, sc.goals, titles[m], os.path.join(idir, fname))
            if animate:
                animate_comparison(grid, sc, ok, titles, os.path.join(idir, "animation.gif"))
            tb.image_file(f"comparison/instance_{i:03d}", comp)

    summary = _aggregate(scenarios[0].num_agents, per_instance, methods, times, fails)
    summary["expert_name"] = ename
    table = format_table([summary])
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"{label}\n\n{header}\n\n{table}\n\n"
                + _per_instance_md(per_instance, methods, names) + "\n")
    _write_csv(os.path.join(out, "summary.csv"), [summary])
    _write_csv(os.path.join(out, "per_instance.csv"), per_instance)
    chart = plot_test_summary(per_instance, methods, os.path.join(out, "summary.png"),
                              f"Delay: {label} (lower is better)\n{header}",
                              expert_name=ename)
    if tb is not None:
        tb.image_file("summary", chart)
        tb.text("summary", f"{label}\n\n{header}\n\n{table}")
        tb.close()
    log("\n" + table)
    log(f"\nResults (folder replaced on every test): {out}")
    log("  summary.png, summary.md, per_instance.csv"
        + (", instance_XXX/comparison.png" if plots else ""))
    return summary


def _aggregate(n, per_instance, methods, times, fails):
    expert = [r["Expert"] for r in per_instance if r["Expert"] is not None]
    e = float(np.mean(expert)) if expert else float("nan")
    row = {"agents": n, "num_test": len(per_instance), "expert_delay": e}
    for m in methods:
        if m == "Expert":
            continue
        key = KEY[m]
        vals = [r[m] for r in per_instance if r[m] is not None]
        d = float(np.mean(vals)) if vals else float("nan")
        row[f"{key}_delay"] = d
        row[f"{key}_gap"] = optimality_gap(d, e) if expert and vals else float("nan")
        if key == "cbs":
            row["cbs_solved"] = len(vals)
            continue
        row[f"{key}_time"] = float(np.mean(times[m])) if times[m] else float("nan")
        if key != "orig":
            row[f"{key}_failures"] = fails[m]
    return row


def evaluate(cfg, agent_counts, models=("edge", "gnn"), log=print):
    """Table-1 numbers for several agent counts (all saved test scenarios, no plots)."""
    return [test_saved(cfg, n, None, models, plots=False, log=log) for n in agent_counts]


def format_table(rows):
    """Markdown table in the layout of the paper's Table 1 (+ optimal CBS)."""
    ename = rows[0].get("expert_name", "EECBS") if rows else "EECBS"
    lines = [
        f"| Agents | PP (orig) | PP (Edge) | PP (GNN) | {ename} | CBS (optimal) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        cells = []
        for m in ("orig", "edge", "gnn"):
            if f"{m}_delay" in r:
                cells.append(f"{r[f'{m}_delay']:.2f} ({r[f'{m}_gap']:.1f}%)")
            else:
                cells.append("—")
        cbs = "—"
        if r.get("cbs_solved"):
            cbs = f"{r['cbs_delay']:.2f}"
            if r["cbs_solved"] < r["num_test"]:
                cbs += f" ({r['cbs_solved']}/{r['num_test']} solved)"
        lines.append(f"| {r['agents']} | " + " | ".join(cells)
                     + f" | {r['expert_delay']:.2f} | {cbs} |")
    lines.append("")
    lines.append(f"Average delay (% optimality gap relative to {ename}) over "
                 f"{', '.join(str(r['num_test']) for r in rows)} instance(s). Lower is better. "
                 "CBS is averaged over the instances it solved within its time limit.")
    timing = []
    for r in rows:
        parts = [f"{m}: {r[f'{m}_time'] * 1000:.1f} ms" for m in ("orig", "edge", "gnn")
                 if f"{m}_time" in r]
        failures = [f"{m} PP failures: {r[f'{m}_failures']}" for m in ("edge", "gnn")
                    if r.get(f"{m}_failures")]
        timing.append(f"- n={r['agents']}: " + ", ".join(parts + failures))
    return "\n".join(lines + ["", "Mean time per instance (model inference + PP):"] + timing)


def _per_instance_md(per_instance, methods, names):
    lines = ["| instance | " + " | ".join(names[m] for m in methods) + " |",
             "|---:|" + "---:|" * len(methods)]
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
