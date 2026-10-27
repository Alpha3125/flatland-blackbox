"""Prioritized Planning with learned edge costs (paper Sec. 3.1 and 3.3).

Agents are planned one at a time in a fixed priority order. Each agent runs
space-time A*, treating vertices and edges reserved by higher-priority agents
as dynamic obstacles (Silver, 2005).

Learned costs only change *which* path is preferred:
  * the A* g-value accumulates the learned weight w_e of every traversed
    directed edge (including the self-loop for a wait);
  * every action still takes exactly one timestep, so the reservation table is
    built with the original unit costs ("we retain the original graph costs to
    determine traversal durations");
  * the heuristic is the true distance to the goal on the obstacle-free graph,
    recomputed under the current learned weights so it stays admissible.

Agents stay at their goal after arriving, so an agent may only finish at time t
if no higher-priority agent occupies its goal at any time >= t, and once it has
finished its goal is blocked for everyone planned after it.
"""

from __future__ import annotations

from heapq import heappop, heappush

import numpy as np


class NoSolutionError(Exception):
    """Raised when PP cannot find a path for some agent."""

    def __init__(self, agent, message=None):
        super().__init__(message or f"PP found no path for agent {agent}")
        self.agent = agent


class ReservationTable:
    """Space-time reservations of already-planned (higher-priority) agents."""

    def __init__(self, num_nodes):
        self.N = num_nodes
        self.vertex = set()  # key t*N + v
        self.edge = set()  # key (t*N + u)*N + v : move u->v between t and t+1
        self.goal_since = {}  # v -> t: v is occupied forever from time t
        self.last_time = {}  # v -> last timestep v is reserved (excluding goal_since)
        self.max_time = 0

    def reserve(self, path):
        N = self.N
        for t, v in enumerate(path):
            self.vertex.add(t * N + v)
            if self.last_time.get(v, -1) < t:
                self.last_time[v] = t
        for t in range(len(path) - 1):
            u, v = path[t], path[t + 1]
            if u != v:
                self.edge.add((t * N + u) * N + v)
        T = len(path) - 1
        self.goal_since[path[-1]] = T
        self.max_time = max(self.max_time, T)


def distance_order(grid, starts, goals):
    """Priority order used in the paper: ascending true distance (ties by index)."""
    d = [grid.bfs_distances(g)[s] for s, g in zip(starts, goals)]
    return sorted(range(len(starts)), key=lambda i: (d[i], i))


class PrioritizedPlanner:
    """PP on a :class:`~mapf_nco.grid.Grid` with optional learned edge weights."""

    def __init__(self, grid, w_min=1e-3):
        self.grid = grid
        self.w_min = float(w_min)
        self._unit_h_cache = {}

    # ------------------------------------------------------------------ public
    def solve(self, starts, goals, order=None, weights=None):
        """Plan all agents.

        Args:
            starts, goals: sequences of node ids, one per agent.
            order: priority order (list of agent indices). Defaults to the
                ascending-distance order of the paper.
            weights: array of shape (num_edges,) with learned costs. ``None``
                means the original uniform cost 1 on every edge.

        Returns:
            list of paths (lists of node ids, one entry per timestep), indexed by
            agent. Each path ends when the agent reaches its goal for good.

        Raises:
            NoSolutionError if some agent cannot be planned.
        """
        n = len(starts)
        if order is None:
            order = distance_order(self.grid, starts, goals)
        if weights is None:
            w = None
            h_all = None
        else:
            w = np.maximum(np.asarray(weights, dtype=np.float64), self.w_min)
            h_all = self.grid.weighted_distances_to([goals[i] for i in order], w)

        res = ReservationTable(self.grid.num_nodes)
        paths = [None] * n
        for k, i in enumerate(order):
            if h_all is None:
                h = self._unit_h(goals[i])
            else:
                h = h_all[k]
            path = self._space_time_astar(starts[i], goals[i], h, w, res)
            if path is None:
                raise NoSolutionError(i)
            res.reserve(path)
            paths[i] = path
        return paths

    # --------------------------------------------------------------- internals
    def _unit_h(self, goal):
        h = self._unit_h_cache.get(goal)
        if h is None:
            h = self.grid.bfs_distances(goal)
            if len(self._unit_h_cache) < 4096:
                self._unit_h_cache[goal] = h
        return h

    def _space_time_astar(self, start, goal, h, w, res):
        N = self.grid.num_nodes
        out_edges = self.grid.out_edges
        vertex_res, edge_res, goal_since = res.vertex, res.edge, res.goal_since
        INF = float("inf")

        if start in vertex_res or goal_since.get(start, INF) <= 0:
            return None
        if h[start] == INF:
            return None

        goal_free_after = res.last_time.get(goal, -1)
        # Beyond this time all reservations are static (goal blocks only), so a
        # longer search cannot find anything new.
        horizon = res.max_time + N + 1

        start_key = start  # t = 0
        g_best = {start_key: 0.0}
        parent = {start_key: -1}
        closed = set()
        counter = 0
        heap = [(float(h[start]), float(h[start]), 0, start, 0)]

        while heap:
            f, hv, _, v, t = heappop(heap)
            key = t * N + v
            if key in closed:
                continue
            closed.add(key)

            if v == goal and t >= goal_free_after:
                path = []
                while key != -1:
                    path.append(key % N)
                    key = parent[key]
                path.reverse()
                return path

            if t >= horizon:
                continue
            g = g_best[key]
            nt = t + 1
            base_next = nt * N
            for u, e in out_edges[v]:
                nkey = base_next + u
                if nkey in vertex_res or nkey in closed:
                    continue
                if goal_since.get(u, INF) <= nt:
                    continue
                # Swap conflict: someone moves u -> v while we move v -> u.
                if u != v and ((t * N + u) * N + v) in edge_res:
                    continue
                hu = h[u]
                if hu == INF:
                    continue
                ng = g + (1.0 if w is None else w[e])
                if ng < g_best.get(nkey, INF):
                    g_best[nkey] = ng
                    parent[nkey] = key
                    counter += 1
                    heappush(heap, (ng + hu, float(hu), counter, u, nt))
        return None


def solve_pp(grid, starts, goals, order=None, weights=None, w_min=1e-3):
    """Convenience wrapper around :class:`PrioritizedPlanner`."""
    return PrioritizedPlanner(grid, w_min=w_min).solve(starts, goals, order, weights)
