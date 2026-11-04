"""Potential-based edge costs (PotP): map features and the planner setting."""

import numpy as np

from mapf_nco.grid import Grid
from mapf_nco.planners import kinds_of, model_ids, training_runs
from mapf_nco.potential import (STRENGTHS, feature_matrix, init_theta, obstacle_distance,
                                potential_costs)

# A tiny map: a single-lane corridor (row 1, columns 1..4) into an open room.
TINY = ["######",
        "#....#",
        "####.#",
        "#....#",
        "#....#",
        "######"]


def _tiny(tmp_path):
    p = tmp_path / "tiny.map"
    p.write_text("type octile\nheight 6\nwidth 6\nmap\n" + "\n".join(TINY) + "\n")
    return Grid.from_file(str(p))


def _edge(g, a, b):
    return int(g.edge_matrix[g.node_of[a], g.node_of[b]])


def test_obstacle_distance_and_near(tmp_path):
    g = _tiny(tmp_path)
    d = obstacle_distance(g)
    assert d.min() == 1  # every free cell of this map touches a wall
    F = feature_matrix(g)
    e = _edge(g, (3, 1), (3, 2))
    assert np.isclose(F[e, STRENGTHS.index("near")], 1.0 / d[g.node_of[(3, 2)]])
    assert (F[:, STRENGTHS.index("toward")] >= 0).all()


def test_near_and_toward_follow_the_move_direction():
    """On the real map: moving from 2 steps away to next to an obstruction climbs the potential."""
    g = Grid.from_file("data/random-32-32-20.map")
    d, F = obstacle_distance(g), feature_matrix(g)
    i_near, i_toward = STRENGTHS.index("near"), STRENGTHS.index("toward")
    e = next(e for e in g.move_edges if d[g.edge_src[e]] == 2 and d[g.edge_dst[e]] == 1)
    back = _edge(g, tuple(g.coords[g.edge_dst[e]]), tuple(g.coords[g.edge_src[e]]))
    assert F[e, i_near] == 1.0 and F[back, i_near] == 0.5  # near is about the cell moved INTO
    assert F[e, i_toward] == 0.5 and F[back, i_toward] == 0.0  # only approaching costs extra


def test_corridor_dead_end_and_lanes(tmp_path):
    g = _tiny(tmp_path)
    F = feature_matrix(g)
    k = len(STRENGTHS)
    into_corridor = _edge(g, (1, 2), (1, 3))  # walls above and below (1, 3)
    into_room = _edge(g, (3, 1), (3, 2))  # (2, 2) is a wall but (4, 2) is free
    assert F[into_corridor, STRENGTHS.index("corridor")] == 1
    assert F[into_room, STRENGTHS.index("corridor")] == 0
    dead = _edge(g, (1, 2), (1, 1))  # (1, 1) has one free neighbour
    assert F[dead, STRENGTHS.index("dead_end")] == 1
    # east in row 1 -> +1 in row 1's lane column; west -> -1
    assert F[into_corridor, k + 1] == 1 and F[_edge(g, (1, 3), (1, 2)), k + 1] == -1
    down = _edge(g, (1, 4), (2, 4))  # south in column 4
    assert F[down, k + g.height + 4] == 1


def test_waits_and_zero_init_keep_unit_costs():
    g = Grid.from_file("data/random-32-32-20.map")
    F = feature_matrix(g)
    assert not F[g.is_self_loop].any()
    assert ((F[g.move_edges, len(STRENGTHS):] != 0).sum(1) == 1).all()  # one lane per move
    assert np.allclose(potential_costs(g, init_theta(g), F), 1.0)
    w = potential_costs(g, init_theta(g, {"toward": 0.2, "corridor": 0.2}), F)
    assert (w >= 1).all() and (w > 1).any()


def test_unknown_init_key_is_an_error():
    import pytest

    g = Grid.from_file("data/random-32-32-20.map")
    with pytest.raises(ValueError):
        init_theta(g, {"towards": 0.5})  # typo of "toward"


def test_alternating_lanes(tmp_path):
    g = _tiny(tmp_path)
    w = potential_costs(g, init_theta(g, {"lanes": 0.2}))
    east, west = _edge(g, (4, 1), (4, 2)), _edge(g, (4, 2), (4, 1))  # row 4 (even): east cheaper
    assert np.isclose(w[east], 0.8) and np.isclose(w[west], 1.2)
    south, north = _edge(g, (3, 1), (4, 1)), _edge(g, (4, 1), (3, 1))  # column 1 (odd): north cheaper
    assert np.isclose(w[south], 1.2) and np.isclose(w[north], 0.8)


def test_model_setting():
    assert kinds_of("potential_edge") == ["potential", "potential_edge"]
    assert training_runs({"model": "potential_edge", "planner": "both"}) == [
        ("potential", "pp"), ("potential_edge", "pp"),
        ("potential", "cbs"), ("potential_edge", "cbs")]
    assert model_ids({"model": "potential", "planner": "cbs"}) == ["potential_cbs"]
    assert model_ids({}) == ["edge", "edge_cbs"]


def test_main_lists_every_model_id():
    import main
    from mapf_nco.planners import MODEL_IDS

    assert tuple(main.MODEL_ID_CHOICES) == MODEL_IDS
