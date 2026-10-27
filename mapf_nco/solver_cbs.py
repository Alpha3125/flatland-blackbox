"""Pure-Python Conflict-Based Search (Sharon et al., 2015) — fallback expert.

The paper uses EECBS (C++) to produce expert plans. This optimal CBS exists so
the pipeline can be exercised end-to-end without compiling EECBS. It is fine for
small agent counts but will usually hit its time limit at 50-100 agents on
random-32-32-20; use :mod:`mapf_nco.solver_eecbs` for the paper's experiments.
"""

from __future__ import annotations

import time
from heapq import heappop, heappush

from mapf_nco.metrics import find_conflicts


def _low_level(grid, start, goal, vcons, econs, max_t):
    """Unit-cost space-time A* under CBS constraints.

    vcons: set of (v, t); econs: set of (u, v, t) meaning move u->v arriving at t.
    The agent may only finish at time t if its goal has no vertex constraint at
    any time >= t.
    """
    h = grid.bfs_distances(goal)
    if h[start] == float("inf") or (start, 0) in vcons:
        return None
    last_goal_con = max((t for (v, t) in vcons if v == goal), default=-1)
    horizon = max(max_t, last_goal_con) + grid.num_nodes + 1
    heap = [(h[start], h[start], 0, start)]
    parent = {(start, 0): None}
    closed = set()
    while heap:
        _, _, t, v = heappop(heap)
        if (v, t) in closed:
            continue
        closed.add((v, t))
        if v == goal and t > last_goal_con:
            path, s = [], (v, t)
            while s is not None:
                path.append(s[0])
                s = parent[s]
            return path[::-1]
        if t >= horizon:
            continue
        nt = t + 1
        for u in grid.neighbors[v] + [v]:
            s = (u, nt)
            if s in parent or (u, nt) in vcons or (v, u, nt) in econs:
                continue
            parent[s] = (v, t)
            heappush(heap, (nt + h[u], h[u], nt, u))
    return None


def solve_cbs(grid, starts, goals, time_limit=60.0, max_nodes=200000):
    """Optimal CBS. Returns a list of paths or None if it times out."""
    t0 = time.time()
    n = len(starts)
    cons = [(frozenset(), frozenset()) for _ in range(n)]
    paths = []
    for i in range(n):
        p = _low_level(grid, starts[i], goals[i], set(), set(), 0)
        if p is None:
            return None
        paths.append(p)

    def cost(ps):
        return sum(len(p) - 1 for p in ps)

    counter = 0
    open_list = [(cost(paths), 0, counter, cons, paths)]
    expanded = 0
    while open_list:
        if time.time() - t0 > time_limit or expanded >= max_nodes:
            return None
        _, _, _, cons, paths = heappop(open_list)
        expanded += 1
        conflicts = find_conflicts(paths, max_conflicts=1)
        if not conflicts:
            return paths
        kind, t, i, j, where = conflicts[0]
        for agent in (i, j):
            vc, ec = cons[agent]
            if kind == "vertex":
                vc = vc | {(where, t)}
            else:
                u, v = where if agent == i else (where[1], where[0])
                ec = ec | {(u, v, t)}
            max_t = max([tt for (_, tt) in vc] + [tt for (_, _, tt) in ec] + [0])
            p = _low_level(grid, starts[agent], goals[agent], vc, ec, max_t)
            if p is None:
                continue
            new_cons = list(cons)
            new_cons[agent] = (vc, ec)
            new_paths = list(paths)
            new_paths[agent] = p
            counter += 1
            n_conf = len(find_conflicts(new_paths, max_conflicts=10))
            heappush(open_list, (cost(new_paths), n_conf, counter, new_cons, new_paths))
    return None
