"""CBS as the downstream planner of the learned costs (WeightedCBS)."""

import itertools

import numpy as np

from mapf_nco.blackbox import blackbox_step
from mapf_nco.metrics import edge_usage, sum_of_costs, validate_solution
from mapf_nco.planners import model_id, model_ids, planner_of, planners_of
from mapf_nco.scenarios import assign_priority_order, make_rng, random_scenario
from mapf_nco.solver_cbs import WeightedCBS, solve_cbs
from mapf_nco.solver_pp import NoSolutionError, PrioritizedPlanner


def _cost(grid, paths, w):
    return float(edge_usage(grid, paths) @ w)


def test_unit_costs_match_optimal_cbs(grid):
    cbs = WeightedCBS(grid, time_limit=30)
    for seed in range(3):
        sc = random_scenario(grid, 8, make_rng(seed))
        ref = solve_cbs(grid, sc.starts, sc.goals, time_limit=30)
        for w in (None, np.ones(grid.num_edges)):
            paths = cbs.solve(sc.starts, sc.goals, None, w)
            validate_solution(grid, paths, sc.starts, sc.goals)
            assert sum_of_costs(paths) == sum_of_costs(ref)


def test_weighted_plan_is_feasible_and_cheapest(grid):
    """No PP plan (any priority order) is cheaper under the learned costs."""
    rng = np.random.default_rng(0)
    cbs = WeightedCBS(grid, w_min=1e-3, time_limit=30)
    pp = PrioritizedPlanner(grid, w_min=1e-3)
    for seed in range(4):
        sc = random_scenario(grid, 4, make_rng(seed))
        w = rng.uniform(0.2, 3.0, grid.num_edges)
        paths = cbs.solve(sc.starts, sc.goals, None, w)
        validate_solution(grid, paths, sc.starts, sc.goals)
        best = _cost(grid, paths, w)
        for order in itertools.permutations(range(4)):
            try:
                other = pp.solve(sc.starts, sc.goals, list(order), w)
            except NoSolutionError:
                continue
            assert best <= _cost(grid, other, w) + 1e-9


def test_agent_offsets_change_only_that_agent(grid):
    cbs = WeightedCBS(grid, time_limit=30)
    sc = random_scenario(grid, 1, make_rng(3))
    (p0,) = cbs.solve(sc.starts, sc.goals)
    off = np.zeros(grid.num_edges)
    off[np.isin(grid.edge_dst, p0[1:-1])] = 5.0  # make its current route expensive
    (p1,) = cbs.solve(sc.starts, sc.goals, None, None, [off])
    assert _cost(grid, [p1], 1 + off) <= _cost(grid, [p0], 1 + off)


def test_timeout_raises():
    from mapf_nco.grid import Grid

    g = Grid.from_file("data/random-32-32-20.map")
    sc = random_scenario(g, 60, make_rng(1))
    try:
        WeightedCBS(g, time_limit=0.0).solve(sc.starts, sc.goals)
    except NoSolutionError:
        return
    raise AssertionError("expected a timeout")


def test_blackbox_through_cbs_moves_towards_expert(grid):
    """The Pogancic gradient is non-zero and points the costs towards the expert's edges."""
    cbs = WeightedCBS(grid, time_limit=30)
    found = False
    for seed in range(6):
        sc = random_scenario(grid, 6, make_rng(seed))
        assign_priority_order(grid, sc, make_rng(seed))
        expert = PrioritizedPlanner(grid).solve(sc.starts, sc.goals, sc.order)  # a different plan
        y_star = edge_usage(grid, expert)
        w = np.ones(grid.num_edges)
        r = blackbox_step(cbs, sc.starts, sc.goals, sc.order, w, y_star, lam=5.0)
        assert r.ok
        if np.any(r.grad):
            found = True
            # gradient descent never raises the cost of an edge only the expert uses
            y_hat = edge_usage(grid, cbs.solve(sc.starts, sc.goals, None, w))
            only_expert = (y_star > 0) & (y_hat == 0)
            assert np.all(r.grad[only_expert] >= 0)
    assert found


def test_planner_setting():
    assert planners_of("both") == ["pp", "cbs"]
    assert model_ids({"planner": "cbs"}) == ["edge_cbs"]
    assert model_ids({"planner": "both"}) == ["edge", "edge_cbs"]
    assert model_ids({}) == ["edge", "edge_cbs"]  # default: double train
    assert model_id("pp") == "edge" and model_id("cbs") == "edge_cbs"
    assert planner_of("edge_cbs") == "cbs" and planner_of("edge") == "pp"
