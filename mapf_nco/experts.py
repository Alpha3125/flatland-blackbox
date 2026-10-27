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
    assign_priority_order,
    dataset_path,
    make_rng,
    random_scenario,
    save_dataset,
)
from mapf_nco.solver_cbs import solve_cbs
from mapf_nco.solver_eecbs import (check_eecbs, explain_missing, find_eecbs, solve_eecbs,
                                   uses_wsl)

_G = None


def _init_worker(map_path):
    global _G
    _G = Grid.from_file(map_path)


def _solve_task(task):
    """Worker: sample one scenario, fix its PP order, solve with the expert."""
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
    return sc.to_dict()


def suboptimality_for(n, cfg):
    """Paper: bound 1.0 for 50 agents and 1.05 for the harder 75/100-agent sets."""
    table = {int(k): float(v) for k, v in cfg.get("suboptimality", {}).items()}
    return table.get(n, cfg.get("default_suboptimality", 1.05))


def generate_datasets(cfg, agent_counts, workers=None, log=print):
    """Create ``train`` and ``test`` datasets for each agent count."""
    from mapf_nco.scenarios import Scenario

    grid = Grid.from_file(cfg["map_path"])
    expert = cfg.get("expert", "eecbs")
    opts = {"time_limit": float(cfg.get("expert_time_limit", 60.0))}
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
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    for n in agent_counts:
        opts["suboptimality"] = suboptimality_for(n, cfg)
        for split, size, seed0 in (
            ("train", cfg["num_train"], cfg.get("seed", 0) * 1_000_000 + n * 10_000),
            ("test", cfg["num_test"], cfg.get("seed", 0) * 1_000_000 + n * 10_000 + 5_000),
        ):
            out = dataset_path(cfg["data_dir"], grid.name, n, split)
            if os.path.exists(out) and not cfg.get("overwrite", False):
                log(f"[generate] {out} exists, skipping (use --overwrite to redo)")
                continue
            label = f"eecbs w={opts['suboptimality']}" if expert == "eecbs" else "python cbs"
            log(f"[generate] n={n} {split}: {size} scenarios with {label} ({workers} workers)")
            results, next_seed, attempts = [], seed0, 0
            t0 = time.time()
            with ProcessPoolExecutor(workers, initializer=_init_worker,
                                     initargs=(cfg["map_path"],)) as pool, \
                    tqdm(total=size, desc=f"n={n} {split}", unit="scen") as bar:
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
            log(f"  solved {len(results)}/{size} (attempts {attempts}, {time.time() - t0:.0f}s)")
            scenarios = [Scenario.from_dict(d) for d in results]
            save_dataset(out, grid, scenarios, info={
                "num_agents": n, "split": split, "expert": expert,
                "suboptimality": opts["suboptimality"],
                "time_limit": opts["time_limit"], "attempts": attempts,
            })
            log(f"  saved {out}")
