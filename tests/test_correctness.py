"""Independent correctness checks of the algorithms.

Each test compares the real implementation with a deliberately simple,
separately written reference (brute force / explicit graph search) on small
instances, or checks an invariant that must hold by definition.
"""

import heapq
import itertools
import random

import numpy as np
import pytest

from mapf_nco.blackbox import blackbox_step
from mapf_nco.grid import Grid
from mapf_nco.metrics import edge_usage, find_conflicts, lower_bound, sum_of_costs
from mapf_nco.scenarios import assign_priority_order, make_rng, random_scenario
from mapf_nco.solver_cbs import solve_cbs
from mapf_nco.solver_pp import PrioritizedPlanner, distance_order

SMALL_MAP = [
    "......@.",
    ".@@...@.",
    "........",
    "..@.@@..",
    "..@.....",
    "....@...",
    ".@......",
    "...@....",
]


@pytest.fixture
def small():
    return Grid(SMALL_MAP, name="small8")


def _random_agents(g, n, seed):
    cells = random.Random(seed).sample(g.largest_component(), 2 * n)
    return cells[:n], cells[n:]


def _path_cost(g, path, w):
    return sum(w[g.edge_matrix[u, v]] for u, v in zip(path[:-1], path[1:]))


# ------------------------------------------------------------------- grid
def test_grid_matches_raw_map(grid):
    """Edges = exactly the 4-neighbour pairs of '.' cells, read straight from the file."""
    rows = grid.rows
    free = {(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "."}
    expected = {(a, b) for a in free for b in free
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1}
    got = {(grid.cell(u), grid.cell(v)) for u, v in zip(grid.edge_src, grid.edge_dst) if u != v}
    assert got == expected
    assert grid.num_nodes == len(free)


# --------------------------------------------------------------------- PP
def _reference_space_time_dijkstra(g, start, goal, w, higher_paths):
    """Plain Dijkstra over explicit (cell, time) states, no heuristic.

    Constraints from the higher-priority paths, written out independently:
      - vertex: cannot be where a higher agent is at the same time
        (higher agents stay at their goal forever after their path ends);
      - swap: cannot traverse u->v while a higher agent traverses v->u;
      - finish: may stop at the goal at time t only if no higher agent is at the
        goal at any time >= t.
    Returns the minimum learned cost to reach the goal for good.
    """
    T_end = max([len(p) - 1 for p in higher_paths] + [0])
    horizon = T_end + g.num_nodes + 1

    def pos(p, t):
        return p[min(t, len(p) - 1)]

    def occupied(v, t):
        return any(pos(p, t) == v for p in higher_paths)

    def swapped(u, v, t):  # move u(t) -> v(t+1)
        return any(pos(p, t) == v and pos(p, t + 1) == u for p in higher_paths)

    def can_finish(t):
        return not any(pos(p, tt) == goal for p in higher_paths for tt in range(t, T_end + 1))

    if occupied(start, 0):
        return None
    dist = {(start, 0): 0.0}
    pq = [(0.0, start, 0)]
    while pq:
        d, v, t = heapq.heappop(pq)
        if d > dist[(v, t)]:
            continue
        if v == goal and can_finish(t):
            return d
        if t >= horizon:
            continue
        for u in g.neighbors[v] + [v]:
            if occupied(u, t + 1) or (u != v and swapped(v, u, t)):
                continue
            nd = d + w[g.edge_matrix[v, u]]
            if nd < dist.get((u, t + 1), float("inf")) - 1e-12:
                dist[(u, t + 1)] = nd
                heapq.heappush(pq, (nd, u, t + 1))
    return None


@pytest.mark.parametrize("seed", range(12))
def test_pp_each_agent_is_optimal_given_higher_priorities(small, seed):
    """Every agent's PP path has the minimum learned cost the reservations allow."""
    g = small
    starts, goals = _random_agents(g, 8, seed)
    w = np.random.default_rng(seed).uniform(0.2, 3.0, g.num_edges)
    order = distance_order(g, starts, goals)
    try:
        paths = PrioritizedPlanner(g).solve(starts, goals, order, w)
    except Exception:
        pytest.skip("PP infeasible for this ordering")
    higher = []
    for i in order:
        ref = _reference_space_time_dijkstra(g, starts[i], goals[i], w, higher)
        assert ref is not None
        assert _path_cost(g, paths[i], w) == pytest.approx(ref, abs=1e-9)
        higher.append(paths[i])
    assert not find_conflicts(paths)


def test_pp_single_agent_cost_is_weighted_shortest_path(grid):
    """Alone, PP's learned cost = static weighted distance (= its heuristic value at the start)."""
    rng = np.random.default_rng(1)
    planner = PrioritizedPlanner(grid)
    for _ in range(5):
        s, t = rng.choice(grid.largest_component(), 2, replace=False)
        w = rng.uniform(0.1, 5.0, grid.num_edges)
        (p,) = planner.solve([int(s)], [int(t)], None, w)
        d = grid.weighted_distances_to([int(t)], w)[0][int(s)]
        assert _path_cost(grid, p, w) == pytest.approx(d)


def test_pp_lower_priorities_do_not_change_higher_ones(grid):
    """Planning only the first k agents gives the same paths for them (PP is sequential)."""
    sc = random_scenario(grid, 40, make_rng(9))
    assert assign_priority_order(grid, sc, make_rng(9))
    planner = PrioritizedPlanner(grid)
    full = planner.solve(sc.starts, sc.goals, sc.order)
    top = sc.order[:15]
    part = planner.solve([sc.starts[i] for i in top], [sc.goals[i] for i in top],
                         list(range(15)))
    assert part == [full[i] for i in top]


# -------------------------------------------------------------------- CBS
def _brute_force_min_soc(g, starts, goals):
    """Exact minimum sum-of-costs by joint-state Dijkstra (tiny instances only).

    State = (positions, finished flags). An agent may 'finish' only on its goal
    and then stays there forever; every step costs the number of unfinished
    agents, so the total is the sum of final arrival times.
    """
    n = len(starts)
    start = (tuple(starts), (False,) * n)
    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, (pos, fin) = heapq.heappop(pq)
        if d > dist[(pos, fin)]:
            continue
        if all(fin):
            return d
        # finishing is free and instantaneous
        for i in range(n):
            if not fin[i] and pos[i] == goals[i]:
                nf = fin[:i] + (True,) + fin[i + 1:]
                if d < dist.get((pos, nf), 1e18):
                    dist[(pos, nf)] = d
                    heapq.heappush(pq, (d, (pos, nf)))
        options = [[pos[i]] if fin[i] else g.neighbors[pos[i]] + [pos[i]] for i in range(n)]
        for nxt in itertools.product(*options):
            if len(set(nxt)) < n:
                continue  # vertex conflict
            if any(nxt[i] == pos[j] and nxt[j] == pos[i] and i != j
                   for i in range(n) for j in range(i + 1, n)):
                continue  # swap conflict
            nd = d + sum(not f for f in fin)
            st = (tuple(nxt), fin)
            if nd < dist.get(st, 1e18):
                dist[st] = nd
                heapq.heappush(pq, (nd, st))
    return None


@pytest.mark.parametrize("seed", range(16))
def test_cbs_matches_brute_force_optimum(seed):
    g = Grid(["....", ".@..", "....", "..@."], name="tiny")  # tight: forces interactions
    starts, goals = _random_agents(g, 3, seed)
    paths = solve_cbs(g, starts, goals, time_limit=30)
    assert paths is not None and not find_conflicts(paths)
    assert sum_of_costs(paths) == _brute_force_min_soc(g, starts, goals)


# ------------------------------------------------------- usage / metrics
def test_usage_sums_to_soc_and_delay_nonnegative(grid):
    """Each timestep of each path uses exactly one directed edge, so sum(y) = SOC."""
    for seed in range(3):
        sc = random_scenario(grid, 50, make_rng(seed))
        if not assign_priority_order(grid, sc, make_rng(seed)):
            continue
        paths = PrioritizedPlanner(grid).solve(sc.starts, sc.goals, sc.order)
        y = edge_usage(grid, paths)
        assert y.sum() == sum_of_costs(paths)
        assert (y >= 0).all() and np.all(y == np.round(y))
        assert sum_of_costs(paths) >= lower_bound(grid, sc.starts, sc.goals)


# --------------------------------------------------------------- blackbox
def test_blackbox_matches_paper_equations(grid):
    """blackbox_step reproduces Eqs. 4-6 computed by hand."""
    sc = random_scenario(grid, 20, make_rng(4))
    assign_priority_order(grid, sc, make_rng(4))
    planner = PrioritizedPlanner(grid)
    rng = np.random.default_rng(0)
    w = rng.uniform(0.8, 1.2, grid.num_edges)
    y_star = edge_usage(grid, PrioritizedPlanner(grid).solve(
        sc.starts, sc.goals, sc.order, rng.uniform(0.5, 2.0, grid.num_edges)))
    lam = 1.0
    r = blackbox_step(planner, sc.starts, sc.goals, sc.order, w, y_star, lam, "mean")

    M = grid.num_edges
    y_hat = edge_usage(grid, planner.solve(sc.starts, sc.goals, sc.order, w))
    loss = np.sum((y_hat - y_star) ** 2) / M  # Eq. 4
    w_prime = w + lam * 2 * (y_hat - y_star) / M  # Eq. 5
    y_lam = edge_usage(grid, planner.solve(sc.starts, sc.goals, sc.order, w_prime))
    grad = -(y_hat - y_lam) / lam  # Eq. 6
    assert r.loss == pytest.approx(loss)
    assert np.array_equal(r.grad, grad)
