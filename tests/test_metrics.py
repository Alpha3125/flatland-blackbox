import numpy as np
import pytest

from mapf_nco.metrics import edge_usage, find_conflicts, optimality_gap


def test_edge_usage_counts_moves_and_waits(open_grid):
    g = open_grid
    a, b = g.node(0, 0), g.node(0, 1)
    y = edge_usage(g, [[a, a, b, a]])
    assert y[g.self_loop[a]] == 1
    assert y[g.edge_matrix[a, b]] == 1 and y[g.edge_matrix[b, a]] == 1
    assert y.sum() == 3


def test_invalid_move_rejected(open_grid):
    g = open_grid
    with pytest.raises(ValueError):
        edge_usage(g, [[g.node(0, 0), g.node(2, 2)]])


def test_conflicts(open_grid):
    g = open_grid
    a, b, c = g.node(0, 0), g.node(0, 1), g.node(0, 2)
    assert find_conflicts([[a, b], [c, b]])[0][0] == "vertex"
    assert find_conflicts([[a, b], [b, a]])[0][0] == "edge"
    # agent 0 finished at b and stays there; agent 1 passes b later
    assert find_conflicts([[a, b], [c, c, b, a]])[0][0] == "vertex"
    assert find_conflicts([[a, b], [c, c]]) == []


def test_gap_matches_paper_table():
    assert optimality_gap(56.20, 23.96) == pytest.approx(134.6, abs=0.1)
    assert optimality_gap(354.20, 161.80) == pytest.approx(118.9, abs=0.1)
