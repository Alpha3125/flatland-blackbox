"""Single-scenario comparisons: original PP vs learned PP (Edge / GNN) vs expert.

Used by ``python main.py solve``. Three ways to pick a scenario:

* no ``--index`` / ``--seed``: print a per-scenario table for the whole split
  and plot the scenario where original PP is furthest behind the expert;
* ``--index i``: scenario i of the train/test dataset;
* ``--seed s``: a brand-new random scenario, solved by the configured expert.
"""

from __future__ import annotations

import os

import numpy as np

from mapf_nco.grid import Grid
from mapf_nco.metrics import sum_of_costs, validate_solution
from mapf_nco.scenarios import (
    assign_priority_order, dataset_path, load_dataset, make_rng, random_scenario,
)
from mapf_nco.solvers.pp import NoSolutionError, PrioritizedPlanner

REF = "PP (orig)"
LABELS = {"edge": "PP (Edge)", "gnn": "PP (GNN)"}


def load_models(cfg, grid, n, kinds, log=print):
    models = {}
    for kind in kinds:
        from mapf_nco.paths import checkpoint_path
        from mapf_nco.train import load_model

        ckpt = checkpoint_path(cfg, grid.name, n, kind)
        if os.path.exists(ckpt):
            model, info = load_model(ckpt, grid)
            models[kind] = model
            log(f"loaded {kind} model ({ckpt}, best epoch {info.get('epoch')})")
        else:
            log(f"no {kind} checkpoint at {ckpt} (train it first); skipping")
    return models


def plan_all(planner, sc, models):
    """Plans of every method for one scenario. A failed method maps to None."""
    return plan_all_timed(planner, sc, models)[0]


def plan_all_timed(planner, sc, models):
    """Like :func:`plan_all`, plus wall-clock seconds per method (inference + PP).

    The original PP runs with explicit unit costs so that it goes through the
    same heuristic computation as the learned models (fair timing, same plan).
    """
    import time

    t0 = time.perf_counter()
    plans = {REF: planner.solve(sc.starts, sc.goals, sc.order,
                                np.ones(planner.grid.num_edges))}
    times = {REF: time.perf_counter() - t0}
    for kind, model in models.items():
        import torch

        from mapf_nco.models import make_batch

        t0 = time.perf_counter()
        with torch.no_grad():
            w = model(make_batch([sc]))[0].double().numpy()
        try:
            plans[LABELS[kind]] = planner.solve(sc.starts, sc.goals, sc.order, w)
        except NoSolutionError:
            plans[LABELS[kind]] = None
        times[LABELS[kind]] = time.perf_counter() - t0
    if sc.expert_paths:
        plans["Expert"] = sc.expert_paths
    return plans, times


def _fresh_scenario(cfg, grid, n, seed, log):
    from mapf_nco.experts import suboptimality_for

    sc = random_scenario(grid, n, make_rng(seed))
    if not assign_priority_order(grid, sc, make_rng(seed)):
        raise RuntimeError("PP fails on this scenario for every tried order; pick another seed")
    expert = cfg.get("expert", "eecbs")
    tl = float(cfg.get("expert_time_limit", 60))
    if expert == "eecbs":
        from mapf_nco.solvers.eecbs import find_eecbs, solve_eecbs

        binary = find_eecbs(cfg.get("eecbs_bin"))
        paths = None if binary is None else solve_eecbs(
            grid, sc.starts, sc.goals, binary, suboptimality_for(n, cfg), tl)
        if binary is None:
            log("EECBS not found: comparing without an expert plan")
    else:
        from mapf_nco.solvers.cbs import solve_cbs

        log(f"solving with Python CBS (limit {tl:.0f}s)...")
        paths = solve_cbs(grid, sc.starts, sc.goals, time_limit=tl)
        if paths is None:
            log("CBS timed out: comparing without an expert plan")
    if paths is not None:
        validate_solution(grid, paths, sc.starts, sc.goals)
    sc.expert_paths = paths
    return sc


def compare(cfg, n, split="test", index=None, seed=None, kinds=("edge", "gnn"),
            animate=False, log=print):
    grid = Grid.from_file(cfg["map_path"])
    planner = PrioritizedPlanner(grid, w_min=cfg["w_min"])
    models = load_models(cfg, grid, n, kinds, log)

    if seed is not None:
        sc = _fresh_scenario(cfg, grid, n, seed, log)
        tag = f"n{n}_seed{seed}"
    else:
        scenarios, _ = load_dataset(dataset_path(cfg["data_dir"], grid.name, n, split), grid)
        if index is None:
            index = _overview(grid, planner, scenarios, models, split, log)
        sc = scenarios[index]
        tag = f"n{n}_{split}{index}"

    lb = sc.lower_bound(grid)
    plans = plan_all(planner, sc, models)
    titles = {}
    log(f"\nScenario {tag}: {n} agents, lower bound {lb}")
    log("Σe ye = total edge usage (Eq. 1) = sum of arrival times; delay = Σe ye - lower bound")
    log(f"{'method':10s} {'Σe ye':>6s} {'delay':>6s} {'makespan':>8s}  changed agents vs PP (orig)")
    ref = plans[REF]
    for name, paths in plans.items():
        if paths is None:
            log(f"{name:10s}   PP failed with these weights")
            continue
        soc = sum_of_costs(paths)
        changed = [i for i, (a, b) in enumerate(zip(ref, paths)) if list(a) != list(b)]
        titles[name] = f"{name}: $\\sum_e y_e$ = {soc},  delay {soc - lb}"
        log(f"{name:10s} {soc:6d} {soc - lb:6d} {max(len(p) - 1 for p in paths):8d}  "
            f"{len(changed)} {changed[:15]}{' ...' if len(changed) > 15 else ''}")
    for name, paths in plans.items():  # per-agent arrival differences
        if paths is None or name == REF:
            continue
        diffs = [(i, len(ref[i]) - 1, len(paths[i]) - 1) for i in range(n)
                 if len(ref[i]) != len(paths[i])]
        if diffs:
            log(f"  {name}: arrival time changes (agent: orig -> new): "
                + ", ".join(f"{i}: {a}->{b}" for i, a, b in diffs[:12])
                + (" ..." if len(diffs) > 12 else ""))

    from mapf_nco.viz import animate_comparison, plot_comparison, plot_paths

    from mapf_nco.paths import plots_dir

    out_dir = os.path.join(plots_dir(cfg, grid.name, n), tag)
    ok_plans = {k: v for k, v in plans.items() if v is not None}
    out = plot_comparison(grid, sc, ok_plans, titles, os.path.join(out_dir, "comparison.png"), REF)
    log(f"\nside-by-side: {out}")
    for name, paths in ok_plans.items():
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
        plot_paths(grid, paths, sc.starts, sc.goals, titles[name], os.path.join(out_dir, fname))
    log(f"single plots: {out_dir}")
    if animate:
        gif = animate_comparison(grid, sc, ok_plans, titles, os.path.join(out_dir, "animation.gif"))
        log(f"animation:    {gif}")
    return plans


def _overview(grid, planner, scenarios, models, split, log):
    """Per-scenario delays for every method; returns the most interesting index."""
    names = [REF] + [LABELS[k] for k in models] + ["Expert"]
    log(f"\nDelay per {split} scenario (lower is better):")
    log("  idx " + "".join(f"{nm:>11s}" for nm in names))
    gaps = []
    for i, sc in enumerate(scenarios):
        plans = plan_all(planner, sc, models)
        lb = sc.lower_bound(grid)
        row = [None if plans.get(nm) is None else sum_of_costs(plans[nm]) - lb for nm in names]
        log(f"  {i:3d} " + "".join(f"{'fail' if v is None else v:>11}" for v in row))
        gaps.append((row[0] - row[-1]) if row[-1] is not None else 0)
    best = int(np.argmax(gaps))
    log(f"-> plotting scenario {best} (largest gap between original PP and the expert). "
        f"Use --index to pick another.")
    return best
