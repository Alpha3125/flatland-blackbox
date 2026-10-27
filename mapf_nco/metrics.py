"""Edge-usage vectors and the evaluation metrics of the paper."""

from __future__ import annotations

import numpy as np


def edge_usage(grid, paths):
    """Directed edge-usage vector y (paper Eq. 1).

    y_e counts how many times directed edge e (a move u->v or a wait self-loop
    v->v) is traversed, summed over agents and timesteps. Each path is counted
    until the agent's final arrival at its goal.
    """
    y = np.zeros(grid.num_edges, dtype=np.float64)
    for p in paths:
        if p is None or len(p) < 2:
            continue
        p = np.asarray(p, dtype=np.int64)
        e = grid.edge_matrix[p[:-1], p[1:]]
        if (e < 0).any():
            t = int(np.flatnonzero(e < 0)[0])
            raise ValueError(f"Invalid move {grid.cell(p[t])} -> {grid.cell(p[t + 1])}")
        np.add.at(y, e, 1.0)
    return y


def sum_of_costs(paths):
    """Sum over agents of the arrival time at the goal."""
    return int(sum(len(p) - 1 for p in paths))


def lower_bound(grid, starts, goals):
    """Collision-free lower bound: sum of individual shortest-path lengths."""
    return int(sum(grid.bfs_distances(g)[s] for s, g in zip(starts, goals)))


def delay(grid, paths, starts, goals):
    """Paper's delay: sum-of-costs minus the collision-free lower bound."""
    return sum_of_costs(paths) - lower_bound(grid, starts, goals)


def optimality_gap(method_delay, expert_delay):
    """Relative gap in percent, as in Table 1: (d - d_EECBS) / d_EECBS * 100."""
    if expert_delay == 0:
        return 0.0 if method_delay == 0 else float("inf")
    return 100.0 * (method_delay - expert_delay) / expert_delay


def find_conflicts(paths, max_conflicts=1):
    """Return up to ``max_conflicts`` vertex/swap conflicts.

    Agents stay at their goal after their path ends.
    """
    paths = [list(p) for p in paths]
    T = max(len(p) for p in paths)
    out = []

    def at(p, t):
        return p[t] if t < len(p) else p[-1]

    for t in range(T):
        seen = {}
        for i, p in enumerate(paths):
            v = at(p, t)
            if v in seen:
                out.append(("vertex", t, seen[v], i, v))
                if len(out) >= max_conflicts:
                    return out
            else:
                seen[v] = i
        if t == 0:
            continue
        moves = {}
        for i, p in enumerate(paths):
            u, v = at(p, t - 1), at(p, t)
            if u != v:
                moves[(u, v)] = i
        for (u, v), i in moves.items():
            j = moves.get((v, u))
            if j is not None and i < j:
                out.append(("edge", t, i, j, (u, v)))
                if len(out) >= max_conflicts:
                    return out
    return out


def validate_solution(grid, paths, starts, goals):
    """Raise ValueError if ``paths`` is not a feasible joint plan."""
    if len(paths) != len(starts):
        raise ValueError("Wrong number of paths")
    for i, (p, s, g) in enumerate(zip(paths, starts, goals)):
        if p[0] != s or p[-1] != g:
            raise ValueError(f"Agent {i}: path does not go from start to goal")
    edge_usage(grid, paths)  # checks every step is a legal move or wait
    conflicts = find_conflicts(paths)
    if conflicts:
        raise ValueError(f"Conflict: {conflicts[0]}")
    return True
