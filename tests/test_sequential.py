"""Sequential (Gelato-style) cost prediction: planner, features and training signals."""

import numpy as np

from mapf_nco.metrics import edge_usage, find_conflicts
from mapf_nco.scenarios import assign_priority_order, make_rng, random_scenario
from mapf_nco.sequential import (SEQ_DIM, ReservationFeatures, seq_blackbox_signal,
                                 seq_teacher_signal, sequential_pp)
from mapf_nco.solver_cbs import solve_cbs
from mapf_nco.solver_pp import PrioritizedPlanner


def _scenario(grid, n=20, seed=3):
    sc = random_scenario(grid, n, make_rng(seed))
    assert assign_priority_order(grid, sc, make_rng(seed))
    return sc


def test_constant_costs_reproduce_ordinary_pp(grid):
    sc = _scenario(grid)
    planner = PrioritizedPlanner(grid)
    seen = []

    def ones(x):
        seen.append(x)
        return np.ones(grid.num_edges)

    paths, W, X = sequential_pp(planner, sc.starts, sc.goals, sc.order, ones)
    assert paths == planner.solve(sc.starts, sc.goals, sc.order)
    assert len(seen) == sc.num_agents  # the model is called once per agent
    assert all(x.shape == (grid.num_nodes, SEQ_DIM) for x in X)


def test_features_describe_the_reservations(grid):
    sc = _scenario(grid)
    planner = PrioritizedPlanner(grid)
    paths, _, X = sequential_pp(planner, sc.starts, sc.goals, sc.order,
                                lambda x: np.ones(grid.num_edges))
    first = sc.order[0]
    assert X[first][:, :4].sum() == 0  # nothing reserved before the first agent
    feats = ReservationFeatures(grid)
    for k, i in enumerate(sc.order):
        assert np.allclose(X[i], feats.current(sc.starts[i], sc.goals[i], k, sc.num_agents))
        assert X[i][sc.starts[i], 5] == 1 and X[i][sc.goals[i], 6] == 1
        feats.add(paths[i])
    last = sc.order[-1]
    assert X[last][:, 0].sum() * 64 == sum(len(paths[j]) for j in sc.order[:-1])


def test_costs_can_depend_on_reservations(grid):
    """A cost function that penalises reserved cells changes later agents' plans, collision-free."""
    sc = _scenario(grid, n=30)
    planner = PrioritizedPlanner(grid)

    def avoid(x):
        return 1.0 + 3.0 * x[grid.edge_dst, 0] * 64 / 10  # dearer where others pass

    paths, W, _ = sequential_pp(planner, sc.starts, sc.goals, sc.order, avoid)
    assert not find_conflicts(paths)
    assert not np.allclose(W[sc.order[0]], W[sc.order[-1]])


def test_teacher_targets_are_always_feasible(grid):
    """With the expert's partial plan reserved, every agent can reach its expert path."""
    sc = _scenario(grid, n=8, seed=1)
    expert = solve_cbs(grid, sc.starts, sc.goals, time_limit=30)
    planner = PrioritizedPlanner(grid)
    loss, X, G, ok, exact = seq_teacher_signal(planner, sc.starts, sc.goals, sc.order,
                                              lambda x: np.ones(grid.num_edges), expert, 1.0,
                                              "sum")
    assert ok and len(X) == len(G) == 8
    # with unit costs most steps already reproduce the expert; mismatches get a gradient
    assert 0 <= exact <= 8
    for i in range(8):
        if not np.any(G[i]):
            continue
        # descending the gradient makes the expert's path cheaper relative to PP's choice
        y_star = edge_usage(grid, [expert[i]])
        assert G[i] @ y_star >= 0


def test_blackbox_signal_shapes(grid):
    sc = _scenario(grid, n=15)
    planner = PrioritizedPlanner(grid)
    y_star = edge_usage(grid, planner.solve(sc.starts, sc.goals, sc.order,
                                            np.random.default_rng(0).uniform(.5, 2, grid.num_edges)))
    loss, X, G, ok = seq_blackbox_signal(planner, sc.starts, sc.goals, sc.order,
                                         lambda x: np.ones(grid.num_edges), y_star, 1.0, "mean")
    assert ok and len(G) == 15 and np.isfinite(loss)
    assert any(np.any(g) for g in G)
    loss, X, G, ok = seq_blackbox_signal(planner, sc.starts, sc.goals, sc.order,
                                         lambda x: np.ones(grid.num_edges), y_star, 2.0, "mean",
                                         objective="makespan")
    assert ok and np.isfinite(loss)
