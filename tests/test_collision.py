"""Collision-based edge costs: per-scenario features."""

import numpy as np
import pytest

from mapf_nco.collision import FEATURES, collision_costs, init_theta, scenario_features
from mapf_nco.grid import Grid
from mapf_nco.planners import kinds_of

# A 3 x 3 open room inside walls.
MAP = ["#####",
       "#...#",
       "#...#",
       "#...#",
       "#####"]


def _grid(tmp_path):
    p = tmp_path / "room.map"
    p.write_text("type octile\nheight 5\nwidth 5\nmap\n" + "\n".join(MAP) + "\n")
    return Grid.from_file(str(p))


def _edge(g, a, b):
    return int(g.edge_matrix[g.node_of[a], g.node_of[b]])


def test_crossing_agents_make_the_centre_dear(tmp_path):
    """A goes down the middle column, B along the middle row: both reach the centre at t = 1."""
    g = _grid(tmp_path)
    n = g.node_of
    F = scenario_features(g, [n[(1, 2)], n[(2, 1)]], [n[(3, 2)], n[(2, 3)]], [0, 1])
    i_naive, i_clean = FEATURES.index("naive"), FEATURES.index("clean")
    assert not F[g.is_self_loop].any()  # waits keep cost 1
    into_centre = _edge(g, (2, 1), (2, 2))
    assert np.isclose(F[into_centre, i_naive], np.log1p(1)) and F[into_centre, i_clean] == 0
    into_corner = _edge(g, (3, 2), (3, 3))  # nobody goes near (3, 3)
    assert (F[into_corner, :3] == 0).all() and F[into_corner, i_clean] == -1


def test_one_agent_has_no_hotspots(tmp_path):
    g = _grid(tmp_path)
    F = scenario_features(g, [g.node_of[(1, 1)]], [g.node_of[(3, 3)]], [0])
    assert (F[g.move_edges, :3] == 0).all() and (F[g.move_edges, 3] == -1).all()


def test_costs_and_init():
    g = Grid.from_file("data/random-32-32-20.map")
    from mapf_nco.scenarios import assign_priority_order, make_rng, random_scenario

    sc = random_scenario(g, 30, make_rng(1))
    assign_priority_order(g, sc, make_rng(1))
    F = scenario_features(g, sc.starts, sc.goals, sc.order)
    assert np.allclose(collision_costs(F, init_theta()), 1.0)  # zero strengths = unit costs
    w = collision_costs(F, init_theta({"naive": 0.1, "clean": 0.05}))
    assert (w[g.move_edges][F[g.move_edges, 0] > 0] > 1).all()  # dearer into collision cells
    assert np.allclose(w[g.move_edges][F[g.move_edges, 3] == -1], 0.95)  # cheaper into clean cells
    with pytest.raises(ValueError):
        init_theta({"naïve": 0.1})
    assert kinds_of("collision") == ["collision"]


def test_features_match_a_direct_per_edge_computation():
    """Recompute every move's four features edge by edge from the hotspot counters."""
    from mapf_nco.hotspots import close_calls, gave_way, naive_collisions
    from mapf_nco.scenarios import assign_priority_order, make_rng, random_scenario
    from mapf_nco.solver_pp import PrioritizedPlanner

    g = Grid.from_file("data/random-32-32-20.map")
    for seed in range(6):
        sc = random_scenario(g, 40, make_rng(900 + seed))
        assert assign_priority_order(g, sc, make_rng(900 + seed))
        plan = PrioritizedPlanner(g).solve(sc.starts, sc.goals, sc.order)
        F = scenario_features(g, sc.starts, sc.goals, sc.order)
        naive, gw, cc = (naive_collisions(g, sc.starts, sc.goals)[0], gave_way(plan)[0],
                         close_calls(plan)[0])
        for e in range(g.num_edges):
            v = int(g.edge_dst[e])
            if g.is_self_loop[e]:
                assert not F[e].any()
                continue
            want = [np.log1p(naive[v]), np.log1p(gw[v]), np.log1p(cc[v]),
                    0.0 if (naive[v] or gw[v] or cc[v]) else -1.0]
            assert np.allclose(F[e], want, atol=1e-6), (seed, e)
