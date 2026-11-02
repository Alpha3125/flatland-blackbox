"""Conflict-Based Search (Sharon et al., 2015), optionally with learned edge costs.

Two uses:

* ``solve_cbs``: optimal CBS with unit costs. Fallback expert when EECBS is not
  compiled, and the "CBS (optimal)" reference plans. Fine for small agent
  counts; it usually hits its time limit at 50-100 agents on random-32-32-20.
* ``WeightedCBS``: CBS as the *downstream* planner of the learned models. It
  finds the plan with the smallest total learned cost  sum_e w_e * y_e  (every
  move and wait still takes one timestep). It has the same ``solve`` interface
  as :class:`~mapf_nco.solver_pp.PrioritizedPlanner`, so black-box
  differentiation works through it unchanged. The priority order is ignored.
"""

from __future__ import annotations

import time
from heapq import heappop, heappush

import numpy as np

from mapf_nco.metrics import find_conflicts
from mapf_nco.solver_pp import NoSolutionError


def _low_level(grid, start, goal, vcons, econs, max_t, w=None, h=None):
    """Space-time A* under CBS constraints. Returns (path, cost) or None.

    vcons: set of (v, t); econs: set of (u, v, t) meaning move u->v arriving at t.
    The agent may only finish at time t if its goal has no vertex constraint at
    any time >= t. ``w``: cost of every directed edge (incl. waits; None = 1);
    ``h``: admissible distance-to-goal under ``w``.
    """
    if h is None:
        h = grid.bfs_distances(goal)
    INF = float("inf")
    if h[start] == INF or (start, 0) in vcons:
        return None
    last_goal_con = max((t for (v, t) in vcons if v == goal), default=-1)
    horizon = max(max_t, last_goal_con) + grid.num_nodes + 1
    out_edges = grid.out_edges
    heap = [(float(h[start]), float(h[start]), 0, start)]
    g_best = {(start, 0): 0.0}
    parent = {(start, 0): None}
    closed = set()
    while heap:
        _, _, t, v = heappop(heap)
        if (v, t) in closed:
            continue
        closed.add((v, t))
        if v == goal and t > last_goal_con:
            cost, path, s = g_best[(v, t)], [], (v, t)
            while s is not None:
                path.append(s[0])
                s = parent[s]
            return path[::-1], cost
        if t >= horizon:
            continue
        g, nt = g_best[(v, t)], t + 1
        for u, e in out_edges[v]:
            s = (u, nt)
            if s in closed or s in vcons or (v, u, nt) in econs or h[u] == INF:
                continue
            ng = g + (1.0 if w is None else w[e])
            if ng < g_best.get(s, INF):
                g_best[s] = ng
                parent[s] = (v, t)
                heappush(heap, (ng + h[u], float(h[u]), nt, u))
    return None


def _cbs(grid, starts, goals, ws, hs, time_limit, max_nodes):
    """High-level CBS. ``ws[i]``/``hs[i]``: agent i's costs and heuristic.

    Returns (paths, high-level nodes expanded); paths is None on timeout.
    """
    t0 = time.time()
    n = len(starts)
    cons = [(frozenset(), frozenset()) for _ in range(n)]
    paths, costs = [], []
    for i in range(n):
        r = _low_level(grid, starts[i], goals[i], set(), set(), 0, ws[i], hs[i])
        if r is None:
            return None, 0
        paths.append(r[0])
        costs.append(r[1])
    counter = 0
    open_list = [(sum(costs), 0, counter, cons, paths, costs)]
    expanded = 0
    while open_list:
        if time.time() - t0 > time_limit or expanded >= max_nodes:
            return None, expanded
        _, _, _, cons, paths, costs = heappop(open_list)
        expanded += 1
        conflicts = find_conflicts(paths, max_conflicts=1)
        if not conflicts:
            return paths, expanded
        kind, t, i, j, where = conflicts[0]
        for agent in (i, j):
            vc, ec = cons[agent]
            if kind == "vertex":
                vc = vc | {(where, t)}
            else:
                u, v = where if agent == i else (where[1], where[0])
                ec = ec | {(u, v, t)}
            max_t = max([tt for (_, tt) in vc] + [tt for (_, _, tt) in ec] + [0])
            r = _low_level(grid, starts[agent], goals[agent], vc, ec, max_t, ws[agent],
                           hs[agent])
            if r is None:
                continue
            new_cons = list(cons)
            new_cons[agent] = (vc, ec)
            new_paths, new_costs = list(paths), list(costs)
            new_paths[agent], new_costs[agent] = r
            counter += 1
            n_conf = len(find_conflicts(new_paths, max_conflicts=10))
            heappush(open_list, (sum(new_costs), n_conf, counter, new_cons, new_paths, new_costs))
    return None, expanded


def solve_cbs(grid, starts, goals, time_limit=60.0, max_nodes=200000):
    """Optimal CBS with unit costs. Returns a list of paths or None if it times out."""
    n = len(starts)
    return _cbs(grid, starts, goals, [None] * n, [None] * n, time_limit, max_nodes)[0]


class WeightedCBS:
    """CBS that minimises the total learned edge cost (downstream planner)."""

    def __init__(self, grid, w_min=1e-3, time_limit=10.0, max_nodes=200000):
        self.grid = grid
        self.w_min = float(w_min)
        self.time_limit = float(time_limit)
        self.max_nodes = int(max_nodes)
        self.last_nodes = 0  # high-level nodes expanded by the last solve

    def solve(self, starts, goals, order=None, weights=None, agent_offsets=None):
        """Plan all agents; ``order`` is ignored. Raises NoSolutionError on timeout.

        ``agent_offsets`` as in PrioritizedPlanner.solve: agent i plans with
        ``weights + agent_offsets[i]``.
        """
        n = len(starts)
        grid = self.grid
        if weights is None and agent_offsets is None:
            ws, hs = [None] * n, [None] * n
        else:
            base = (np.ones(grid.num_edges) if weights is None
                    else np.maximum(np.asarray(weights, dtype=np.float64), self.w_min))
            ws, hs, shared = [], [], None
            for i in range(n):
                off = None if agent_offsets is None else agent_offsets[i]
                if off is not None and np.any(off):
                    wi = np.maximum(base + off, self.w_min)
                    hi = grid.weighted_distances_to([goals[i]], wi)[0]
                else:
                    if shared is None:
                        shared = grid.weighted_distances_to(list(goals), base)
                    wi, hi = base, shared[i]
                ws.append(wi)
                hs.append(hi)
        paths, self.last_nodes = _cbs(grid, starts, goals, ws, hs, self.time_limit,
                                      self.max_nodes)
        if paths is None:
            raise NoSolutionError(-1, f"CBS found no plan within {self.time_limit:g} s")
        return paths
