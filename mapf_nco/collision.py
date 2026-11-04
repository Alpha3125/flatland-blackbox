"""Collision-based edge costs: dearer into the cells where collisions happen, cheaper elsewhere.

For one scenario, every directed move u -> v gets four features (waits keep cost 1):

    naive       log(1 + #collisions of the agents' individual shortest paths at v)
    gave_way    log(1 + #waits at v in the original PP plan)
    close_call  log(1 + #close calls at v in the original PP plan)
    clean       -1 if v has none of the three, else 0

(see mapf_nco/hotspots.py). The collision model's costs are

    w = 1 + theta_naive * naive + theta_gave_way * gave_way
          + theta_close_call * close_call + theta_clean * clean

so with positive strengths, moves towards collision hotspots cost more and moves
into clean cells cost less. Unlike the Edge and potential models, the costs
depend on the scenario (its starts, goals and priority order), computed with no
expert: the original PP plan and the individual shortest paths.
"""

from __future__ import annotations

import numpy as np

from mapf_nco.hotspots import close_calls, gave_way, naive_collisions

FEATURES = ("naive", "gave_way", "close_call", "clean")
DEFAULT_INIT = {"naive": 0.0, "gave_way": 0.0, "close_call": 0.0, "clean": 0.0}


def scenario_features(grid, starts, goals, order, pp_paths=None):
    """(M x 4) float32 features of one scenario (see the module docstring).

    ``pp_paths``: the original PP plan (unit costs, this order); planned here if None.
    """
    if pp_paths is None:
        from mapf_nco.solver_pp import PrioritizedPlanner

        pp_paths = PrioritizedPlanner(grid).solve(starts, goals, order)
    counts = np.zeros((3, grid.num_nodes))
    for k, cells in enumerate((naive_collisions(grid, starts, goals)[0], gave_way(pp_paths)[0],
                               close_calls(pp_paths)[0])):
        for v, c in cells.items():
            counts[k, v] = c
    F = np.zeros((grid.num_edges, len(FEATURES)), np.float32)
    moves = grid.move_edges
    dst = grid.edge_dst[moves]
    F[moves, :3] = np.log1p(counts[:, dst]).T
    F[moves, 3] = np.where(counts[:, dst].sum(0) > 0, 0.0, -1.0)
    return F


def init_theta(init=None):
    unknown = set(init or {}) - set(DEFAULT_INIT)
    if unknown:
        raise ValueError(f"unknown collision_init key(s) {sorted(unknown)}; "
                         f"use {', '.join(DEFAULT_INIT)}")
    init = {**DEFAULT_INIT, **(init or {})}
    return np.array([float(init[k]) for k in FEATURES], np.float32)


def collision_costs(F, theta):
    return 1.0 + F.astype(np.float64) @ np.asarray(theta, np.float64)


def describe(theta):
    return ", ".join(f"{n} {float(t):+.3f}" for n, t in zip(FEATURES, theta))
