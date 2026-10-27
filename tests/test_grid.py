import numpy as np


def test_random_map_structure(grid):
    assert (grid.height, grid.width) == (32, 32)
    assert grid.num_nodes == int(grid.free.sum())
    # one self-loop per vertex, the rest are directed moves (two per undirected edge)
    assert grid.is_self_loop.sum() == grid.num_nodes
    assert (~grid.is_self_loop).sum() == grid.degree.sum()
    assert grid.num_edges == grid.num_nodes + grid.degree.sum()


def test_edge_indexing_consistent(grid):
    e = np.arange(grid.num_edges)
    assert np.array_equal(grid.edge_matrix[grid.edge_src, grid.edge_dst], e)
    for u in range(0, grid.num_nodes, 37):
        assert grid.edge_matrix[u, u] == grid.self_loop[u]
        for v, eid in grid.out_edges[u]:
            assert grid.edge_src[eid] == u and grid.edge_dst[eid] == v


def test_weighted_distance_matches_bfs_for_unit_costs(grid):
    targets = [0, 100, grid.num_nodes - 1]
    d = grid.weighted_distances_to(targets, np.ones(grid.num_edges))
    for k, t in enumerate(targets):
        assert np.allclose(d[k], grid.bfs_distances(t))


def test_directed_weights_respected(open_grid):
    g = open_grid
    w = np.ones(g.num_edges)
    a, b = g.node(0, 0), g.node(0, 1)
    w[g.edge_matrix[a, b]] = 10.0  # a->b expensive, b->a cheap
    d_to_b = g.weighted_distances_to([b], w)[0]
    d_to_a = g.weighted_distances_to([a], w)[0]
    assert d_to_b[a] == 3.0  # detour around
    assert d_to_a[b] == 1.0
