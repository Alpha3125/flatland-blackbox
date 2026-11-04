"""Collision hotspots: where collisions were avoided (every plan itself is collision-free).

Agents stay at their goal after their path ends. Each function returns
(Counter cell -> count, total number of events).

* ``gave_way``: wait actions. An agent never waits unless another agent is in
  its way (a wait only adds cost), so each wait is a collision that was avoided
  by giving way.
* ``close_calls``: an agent enters a cell that a different agent occupied one
  timestep before. A one-step slip by either would have been a collision.
* ``naive_collisions``: the collisions between the agents' individual shortest
  paths (each agent planned alone with unit costs): what would happen if nobody
  coordinated. Vertex collisions count at the cell, swaps at both cells. The
  same for every plan of a scenario.
"""

from __future__ import annotations

from collections import Counter


def _horizon(paths):
    return max(len(p) for p in paths)


def _at(p, t):
    return p[t] if t < len(p) else p[-1]


def gave_way(paths):
    cells = Counter()
    for p in paths:
        for t in range(len(p) - 1):
            if p[t] == p[t + 1]:
                cells[p[t]] += 1
    return cells, sum(cells.values())


def close_calls(paths):
    T = _horizon(paths)
    occ = {}  # (cell, t) -> agent
    for i, p in enumerate(paths):
        for t in range(T):
            occ[(_at(p, t), t)] = i
    cells = Counter()
    for (c, t), i in occ.items():
        j = occ.get((c, t + 1))
        if j is not None and j != i:
            cells[c] += 1
    return cells, sum(cells.values())


def individual_paths(grid, starts, goals):
    """Each agent's shortest path planned alone (unit costs, PP's tie-breaking)."""
    from mapf_nco.solver_pp import PrioritizedPlanner

    pp = PrioritizedPlanner(grid)
    return [pp.solve([s], [g])[0] for s, g in zip(starts, goals)]


def naive_collisions(grid, starts, goals):
    paths = individual_paths(grid, starts, goals)
    T = _horizon(paths)
    cells, events = Counter(), 0
    for t in range(T):
        seen = Counter(_at(p, t) for p in paths)
        for c, k in seen.items():
            if k > 1:
                cells[c] += k - 1
                events += k - 1
        if t == 0:
            continue
        moves = {}
        for p in paths:
            u, v = _at(p, t - 1), _at(p, t)
            if u != v:
                moves[(u, v)] = moves.get((u, v), 0) + 1
        for (u, v), k in moves.items():
            if u < v and (v, u) in moves:  # count each swap once
                n = k * moves[(v, u)]
                cells[u] += n
                cells[v] += n
                events += n
    return cells, events
