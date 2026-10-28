"""Generate train/test scenarios with expert plans (paper Sec. 4, Training Setup).

For every agent count n we sample random start/goal scenarios on the map, fix
their PP priority order, and solve them with the expert (EECBS by default).
Scenarios the expert cannot solve within the time limit are discarded and
replaced, so the final dataset has exactly the requested size.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from tqdm import tqdm

from mapf_nco.grid import Grid
from mapf_nco.metrics import validate_solution
from mapf_nco.scenarios import (
    Scenario,
    assign_priority_order,
    load_dataset,
    dataset_path,
    make_rng,
    random_scenario,
    save_dataset,
)
from mapf_nco.solver_cbs import solve_cbs
from mapf_nco.solver_pp import PrioritizedPlanner
from mapf_nco.solver_eecbs import (check_eecbs, explain_missing, find_eecbs, solve_eecbs,
                                   uses_wsl)

_G = None


def _init_worker(map_path):
    global _G
    _G = Grid.from_file(map_path)


def _solve_task(task):
    """Worker: sample one scenario, fix its PP order, and pre-compute every plan.

    expert (EECBS) -> training target; required, the scenario is dropped if it fails.
    PP             -> original prioritized planning (uniform costs).
    CBS            -> optimal plan (EECBS with suboptimality 1.0, i.e. optimal CBS
                      with EECBS's improvements); optional, None if it times out.
    """
    seed, n, expert, opts = task
    rng = make_rng(seed)
    sc = random_scenario(_G, n, rng)
    if not assign_priority_order(_G, sc, rng):
        return None
    t0 = time.time()
    if expert == "eecbs":
        paths = solve_eecbs(
            _G, sc.starts, sc.goals, opts["eecbs_bin"],
            suboptimality=opts["suboptimality"], time_limit=opts["time_limit"],
        )
    elif expert == "cbs":
        paths = solve_cbs(_G, sc.starts, sc.goals, time_limit=opts["time_limit"])
    else:
        raise ValueError(expert)
    if paths is None:
        return None
    validate_solution(_G, paths, sc.starts, sc.goals)
    sc.expert_paths = paths
    sc.expert_solver = expert if expert == "cbs" else f"eecbs(w={opts['suboptimality']})"
    sc.meta.update(seed=seed, expert_time=time.time() - t0)
    sc.lower_bound(_G)
    _add_pp_cbs(sc, expert, opts)
    return sc.to_dict()


def _add_pp_cbs(sc, expert, opts):
    """Store the original PP plan and the optimal (CBS) plan on a scenario."""
    if sc.pp_paths is None:
        t0 = time.time()
        sc.pp_paths = PrioritizedPlanner(_G).solve(sc.starts, sc.goals, sc.order)
        sc.meta["pp_time"] = time.time() - t0
    if sc.cbs_paths is not None or not opts.get("with_cbs", True):
        return
    if expert == "cbs" or (expert == "eecbs" and float(opts["suboptimality"]) == 1.0):
        sc.cbs_paths, sc.cbs_solver = sc.expert_paths, sc.expert_solver  # already optimal
        sc.meta["cbs_time"] = sc.meta.get("expert_time")
        return
    t0 = time.time()
    if opts.get("eecbs_bin"):
        paths = solve_eecbs(_G, sc.starts, sc.goals, opts["eecbs_bin"],
                            suboptimality=1.0, time_limit=opts["cbs_time_limit"])
        solver = "eecbs(w=1.0)"
    else:
        paths, solver = solve_cbs(_G, sc.starts, sc.goals, time_limit=opts["cbs_time_limit"]), "cbs"
    if paths is not None:
        validate_solution(_G, paths, sc.starts, sc.goals)
        sc.cbs_paths, sc.cbs_solver = paths, solver
    sc.meta["cbs_time"] = time.time() - t0
    sc.meta["cbs_timed_out"] = paths is None


def _augment_task(args):
    d, expert, opts = args
    sc = Scenario.from_dict(d)
    _add_pp_cbs(sc, expert, opts)
    return sc.to_dict()


def suboptimality_for(n, cfg):
    """Paper: bound 1.0 for 50 agents and 1.05 for the harder 75/100-agent sets."""
    table = {int(k): float(v) for k, v in cfg.get("suboptimality", {}).items()}
    return table.get(n, cfg.get("default_suboptimality", 1.05))


def expert_options(cfg, grid, log=print):
    """Expert name and solver options; checks that EECBS actually runs."""
    expert = cfg.get("expert", "eecbs")
    opts = {"time_limit": float(cfg.get("expert_time_limit", 60.0)),
            "cbs_time_limit": float(cfg.get("cbs_time_limit", 60.0)),
            "with_cbs": bool(cfg.get("with_cbs", True))}
    if expert == "eecbs":
        binary = find_eecbs(cfg.get("eecbs_bin"))
        if binary is None:
            raise FileNotFoundError(
                explain_missing(cfg.get("eecbs_bin")) + "\nOr pass --eecbs_bin / set "
                "MAPF_NCO_EECBS. For small test runs only, --expert cbs uses Python CBS.")
        ok, msg = check_eecbs(grid, binary)
        if not ok:
            raise RuntimeError(msg)
        log(f"[generate] {msg}{' (via WSL)' if uses_wsl(binary) else ''}")
        opts["eecbs_bin"] = binary
    return expert, opts


def make_scenarios(cfg, n, size, seed0, expert, opts, workers, desc="scenarios"):
    """``size`` random scenarios with EECBS, PP and CBS plans (failed ones replaced)."""
    opts = dict(opts, suboptimality=suboptimality_for(n, cfg))
    results, next_seed = [], seed0
    with ProcessPoolExecutor(workers, initializer=_init_worker,
                             initargs=(cfg["map_path"],)) as pool, \
            tqdm(total=size, desc=desc, unit="scen") as bar:
        pending = set()
        for _ in range(size):  # one task per wanted scenario
            pending.add(pool.submit(_solve_task, (next_seed, n, expert, opts)))
            next_seed += 1
        attempts = size
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                r = fut.result()
                if r is not None:
                    results.append(r)
                    bar.update(1)
                else:  # expert failed (time limit): replace with a new scenario
                    if attempts >= 20 * size:
                        raise RuntimeError("Expert fails on too many scenarios; "
                                           "raise the time limit or suboptimality.")
                    pending.add(pool.submit(_solve_task, (next_seed, n, expert, opts)))
                    next_seed += 1
                    attempts += 1
                bar.set_postfix(attempts=attempts, failed=attempts - size)
    results.sort(key=lambda d: d["meta"]["seed"])  # deterministic order
    scenarios = [Scenario.from_dict(d) for d in results]
    return scenarios, attempts, opts["suboptimality"]


def _cbs_summary(scenarios):
    have = sum(s.cbs_paths is not None for s in scenarios)
    return f"CBS (optimal) plans: {have}/{len(scenarios)}" + (
        "" if have == len(scenarios) else " (the rest timed out; raise cbs_time_limit)")


def generate_datasets(cfg, agent_counts, workers=None, log=print):
    """Create ``train`` and ``test`` datasets for each agent count.

    With ``cfg["augment"]`` an existing dataset is kept and only the missing PP
    and CBS plans are added to it (EECBS plans are not recomputed).
    """
    grid = Grid.from_file(cfg["map_path"])
    expert, opts = expert_options(cfg, grid, log)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    for n in agent_counts:
        for split, size, seed0 in (
            ("train", cfg["num_train"], cfg.get("seed", 0) * 1_000_000 + n * 10_000),
            ("test", cfg["num_test"], cfg.get("seed", 0) * 1_000_000 + n * 10_000 + 5_000),
        ):
            out = dataset_path(cfg["data_dir"], grid.name, n, split)
            if os.path.exists(out) and cfg.get("augment"):
                augment_dataset(cfg, out, grid, expert, opts, workers, log)
                continue
            if os.path.exists(out) and not cfg.get("overwrite", False):
                log(f"[generate] {out} exists, skipping (use --overwrite to redo)")
                continue
            label = (f"eecbs w={suboptimality_for(n, cfg)}" if expert == "eecbs"
                     else "python cbs")
            log(f"[generate] n={n} {split}: {size} scenarios with {label}, plus PP and CBS "
                f"plans ({workers} workers)")
            t0 = time.time()
            scenarios, attempts, w = make_scenarios(cfg, n, size, seed0, expert, opts, workers,
                                                    desc=f"n={n} {split}")
            log(f"  solved {len(scenarios)}/{size} (attempts {attempts}, "
                f"{time.time() - t0:.0f}s); {_cbs_summary(scenarios)}")
            save_dataset(out, grid, scenarios, info={
                "num_agents": n, "split": split, "expert": expert, "suboptimality": w,
                "time_limit": opts["time_limit"], "cbs_time_limit": opts["cbs_time_limit"],
                "attempts": attempts,
            })
            log(f"  saved {out}")


def augment_dataset(cfg, path, grid, expert, opts, workers, log=print):
    """Add missing PP / CBS plans to an existing dataset file."""
    scenarios, info = load_dataset(path, grid)
    todo = [i for i, s in enumerate(scenarios)
            if s.pp_paths is None or (s.cbs_paths is None and not s.meta.get("cbs_timed_out"))]
    if not todo:
        log(f"[generate] {path}: PP and CBS plans already present")
        return
    n = scenarios[0].num_agents
    o = dict(opts, suboptimality=float(info.get("suboptimality", suboptimality_for(n, cfg))))
    log(f"[generate] {path}: adding PP/CBS plans to {len(todo)} scenarios")
    with ProcessPoolExecutor(workers, initializer=_init_worker,
                             initargs=(cfg["map_path"],)) as pool:
        tasks = [(scenarios[i].to_dict(), info.get("expert", expert), o) for i in todo]
        done = list(tqdm(pool.map(_augment_task, tasks), total=len(tasks),
                         desc="adding plans", unit="scen"))
    for i, d in zip(todo, done):
        scenarios[i] = Scenario.from_dict(d)
    info["cbs_time_limit"] = o["cbs_time_limit"]
    save_dataset(path, grid, scenarios, info=info)
    log(f"  {_cbs_summary(scenarios)}; saved {path}")
