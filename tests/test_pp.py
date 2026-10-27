import numpy as np
import pytest

from mapf_nco.metrics import delay, edge_usage, lower_bound, sum_of_costs, validate_solution
from mapf_nco.scenarios import assign_priority_order, make_rng, random_scenario
from mapf_nco.solver_pp import NoSolutionError, PrioritizedPlanner, distance_order, solve_pp


def test_single_agent_is_shortest_path(grid):
    s, g = grid.node(0, 0), grid.node(31, 31)
    (path,) = solve_pp(grid, [s], [g])
    assert len(path) - 1 == grid.bfs_distances(g)[s]


@pytest.mark.parametrize("n", [20, 50])
def test_pp_plans_are_collision_free(grid, n):
    for seed in range(3):
        sc = random_scenario(grid, n, make_rng(seed))
        assert assign_priority_order(grid, sc, make_rng(seed))
        paths = solve_pp(grid, sc.starts, sc.goals, sc.order)
        validate_solution(grid, paths, sc.starts, sc.goals)
        assert sum_of_costs(paths) >= lower_bound(grid, sc.starts, sc.goals)


def test_unit_weights_equal_original(grid):
    sc = random_scenario(grid, 30, make_rng(3))
    assign_priority_order(grid, sc, make_rng(3))
    a = solve_pp(grid, sc.starts, sc.goals, sc.order)
    b = solve_pp(grid, sc.starts, sc.goals, sc.order, weights=np.ones(grid.num_edges))
    assert a == b


def test_deterministic(grid):
    sc = random_scenario(grid, 40, make_rng(4))
    assign_priority_order(grid, sc, make_rng(4))
    w = np.random.default_rng(0).uniform(0.5, 2.0, grid.num_edges)
    assert solve_pp(grid, sc.starts, sc.goals, sc.order, w) == solve_pp(
        grid, sc.starts, sc.goals, sc.order, w)


def test_learned_weights_steer_route_but_not_time(open_grid):
    g = open_grid
    s, t = g.node(2, 0), g.node(2, 4)
    w = np.ones(g.num_edges)
    for c in range(4):  # make the straight row expensive
        w[g.edge_matrix[g.node(2, c), g.node(2, c + 1)]] = 5.0
    (path,) = solve_pp(g, [s], [t], weights=w)
    assert 2 not in {g.cell(v)[0] for v in path[1:-1]}  # avoids row 2
    assert len(path) - 1 == 6  # detour takes real time: durations use unit costs


def test_goal_is_held_after_arrival(open_grid):
    g = open_grid
    # agent 0 parks at (0,2), which lies on agent 1's straight route
    starts = [g.node(0, 1), g.node(0, 0)]
    goals = [g.node(0, 2), g.node(0, 4)]
    paths = solve_pp(g, starts, goals, order=[0, 1])
    validate_solution(g, paths, starts, goals)
    assert g.node(0, 2) not in paths[1]


def test_failure_raises(open_grid):
    from mapf_nco.grid import Grid

    corridor = Grid(["...."], name="corridor")
    # agent 0 parks in the middle of a 1-wide corridor that agent 1 must cross
    starts = [corridor.node(0, 1), corridor.node(0, 0)]
    goals = [corridor.node(0, 2), corridor.node(0, 3)]
    with pytest.raises(NoSolutionError):
        solve_pp(corridor, starts, goals, order=[0, 1])


def test_distance_order(grid):
    sc = random_scenario(grid, 10, make_rng(0))
    order = distance_order(grid, sc.starts, sc.goals)
    d = [grid.bfs_distances(sc.goals[i])[sc.starts[i]] for i in order]
    assert d == sorted(d)
