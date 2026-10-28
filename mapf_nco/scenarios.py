"""Scenarios: sets of n start-goal pairs on a fixed map (paper Sec. 3.4 / 4).

A :class:`Scenario` stores node ids for starts and goals, the PP priority order
used for both training and testing, and (once generated) the expert plan.
Datasets are stored as gzipped JSON so they are portable across machines.
"""

from __future__ import annotations

import gzip
import json
import os
import random
from dataclasses import dataclass, field

import numpy as np

from mapf_nco.metrics import edge_usage, lower_bound, sum_of_costs
from mapf_nco.solver_pp import NoSolutionError, PrioritizedPlanner, distance_order


@dataclass
class Scenario:
    starts: list
    goals: list
    order: list = None
    expert_paths: list = None  # EECBS plan (the training target)
    expert_solver: str = None
    pp_paths: list = None  # original PP plan (uniform costs, this priority order)
    cbs_paths: list = None  # optimal plan (CBS), None if it timed out
    cbs_solver: str = None
    meta: dict = field(default_factory=dict)

    @property
    def num_agents(self):
        return len(self.starts)

    # Cached helpers (not serialised)
    def lower_bound(self, grid):
        if "lower_bound" not in self.meta:
            self.meta["lower_bound"] = lower_bound(grid, self.starts, self.goals)
        return self.meta["lower_bound"]

    def expert_usage(self, grid):
        return edge_usage(grid, self.expert_paths)

    def expert_soc(self):
        return sum_of_costs(self.expert_paths)

    def expert_delay(self, grid):
        return self.expert_soc() - self.lower_bound(grid)

    def to_dict(self):
        return {
            "starts": [int(x) for x in self.starts],
            "goals": [int(x) for x in self.goals],
            "order": None if self.order is None else [int(x) for x in self.order],
            "expert_paths": None
            if self.expert_paths is None
            else [[int(v) for v in p] for p in self.expert_paths],
            "expert_solver": self.expert_solver,
            "pp_paths": _paths(self.pp_paths),
            "cbs_paths": _paths(self.cbs_paths),
            "cbs_solver": self.cbs_solver,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            starts=list(d["starts"]),
            goals=list(d["goals"]),
            order=d.get("order"),
            expert_paths=d.get("expert_paths"),
            expert_solver=d.get("expert_solver"),
            pp_paths=d.get("pp_paths"),
            cbs_paths=d.get("cbs_paths"),
            cbs_solver=d.get("cbs_solver"),
            meta=dict(d.get("meta") or {}),
        )


def _paths(paths):
    return None if paths is None else [[int(v) for v in p] for p in paths]


# ---------------------------------------------------------------- generation
def random_scenario(grid, num_agents, rng, component=None):
    """Distinct random starts and goals inside the largest connected component."""
    component = component if component is not None else grid.largest_component()
    if len(component) < 2 * num_agents:
        raise ValueError("Map too small for that many agents")
    cells = rng.sample(component, 2 * num_agents)
    return Scenario(starts=cells[:num_agents], goals=cells[num_agents:])


def assign_priority_order(grid, scenario, rng, max_random_tries=50):
    """Fix the PP priority order for a scenario, following the paper.

    Agents are ordered by ascending true distance; if PP (with the original
    uniform costs) fails under that order, a single random permutation that
    succeeds is fixed instead. The same order is used for training and testing.

    Returns True if a feasible order was found.
    """
    planner = PrioritizedPlanner(grid)
    order = distance_order(grid, scenario.starts, scenario.goals)
    try:
        planner.solve(scenario.starts, scenario.goals, order)
        scenario.order = order
        scenario.meta["order_type"] = "distance"
        return True
    except NoSolutionError:
        pass
    for _ in range(max_random_tries):
        order = list(range(scenario.num_agents))
        rng.shuffle(order)
        try:
            planner.solve(scenario.starts, scenario.goals, order)
            scenario.order = order
            scenario.meta["order_type"] = "random"
            return True
        except NoSolutionError:
            continue
    return False


# ------------------------------------------------------------------- storage
def save_dataset(path, grid, scenarios, info=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "map": grid.name,
        "num_nodes": grid.num_nodes,
        "num_edges": grid.num_edges,
        "info": info or {},
        "scenarios": [s.to_dict() for s in scenarios],
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def load_dataset(path, grid=None):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    if grid is not None and payload["num_nodes"] != grid.num_nodes:
        raise ValueError(
            f"{path} was generated for map '{payload['map']}' "
            f"({payload['num_nodes']} nodes), not '{grid.name}' ({grid.num_nodes} nodes)"
        )
    return [Scenario.from_dict(d) for d in payload["scenarios"]], payload.get("info", {})


def dataset_path(data_dir, map_name, num_agents, split):
    return os.path.join(data_dir, map_name, f"n{num_agents}", f"{split}.json.gz")


# --------------------------------------------------------- MovingAI .scen I/O
def write_scen(path, grid, starts, goals):
    """Write a MovingAI ``.scen`` file (x = column, y = row), as read by EECBS."""
    map_file = os.path.basename(grid.path) if grid.path else f"{grid.name}.map"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("version 1\n")
        for s, g in zip(starts, goals):
            (sr, sc), (gr, gc) = grid.cell(s), grid.cell(g)
            opt = grid.bfs_distances(g)[s]
            f.write(
                f"0\t{map_file}\t{grid.width}\t{grid.height}\t{sc}\t{sr}\t{gc}\t{gr}\t{opt:.8f}\n"
            )


def read_scen(path, grid, num_agents=None):
    """Read the first ``num_agents`` agents of a MovingAI ``.scen`` file."""
    starts, goals = [], []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < 8:
            continue
        sc, sr, gc, gr = (int(x) for x in parts[4:8])
        starts.append(grid.node(sr, sc))
        goals.append(grid.node(gr, gc))
        if num_agents is not None and len(starts) == num_agents:
            break
    return Scenario(starts=starts, goals=goals)


def make_rng(seed):
    return random.Random(seed)


def train_val_split(scenarios, val_fraction, seed=0):
    """Paper: 20% of the training scenarios are held out for validation."""
    idx = list(range(len(scenarios)))
    random.Random(seed).shuffle(idx)
    n_val = int(round(val_fraction * len(idx)))
    val = [scenarios[i] for i in sorted(idx[:n_val])]
    train = [scenarios[i] for i in sorted(idx[n_val:])]
    return train, val


def scenario_arrays(scenarios):
    """Stack starts/goals/priority ranks into int arrays of shape (B, n)."""
    starts = np.array([s.starts for s in scenarios], dtype=np.int64)
    goals = np.array([s.goals for s in scenarios], dtype=np.int64)
    ranks = np.zeros_like(starts)
    for b, s in enumerate(scenarios):
        for r, i in enumerate(s.order):
            ranks[b, i] = r
    return starts, goals, ranks
