import numpy as np

from mapf_nco.blackbox import blackbox_step, usage_loss
from mapf_nco.metrics import edge_usage
from mapf_nco.solvers.pp import PrioritizedPlanner


def _setup(g):
    s, t = g.node(0, 0), g.node(1, 1)
    via_right = [s, g.node(0, 1), t]
    via_down = [s, g.node(1, 0), t]
    return s, t, via_right, via_down


def test_zero_loss_gives_zero_gradient(open_grid):
    g = open_grid
    planner = PrioritizedPlanner(g)
    s, t, *_ = _setup(g)
    y = edge_usage(g, planner.solve([s], [t]))
    r = blackbox_step(planner, [s], [t], [0], np.ones(g.num_edges), y, lam=1.0)
    assert r.ok and r.loss == 0 and not r.grad.any()


def test_gradient_points_towards_expert(open_grid):
    g = open_grid
    planner = PrioritizedPlanner(g)
    s, t, right, down = _setup(g)
    chosen = planner.solve([s], [t])[0]
    expert = down if chosen == right else right
    y_star = edge_usage(g, [expert])
    for reduction in ("mean", "sum"):
        r = blackbox_step(planner, [s], [t], [0], np.ones(g.num_edges), y_star, 1.0, reduction)
        e_exp = g.edge_matrix[expert[0], expert[1]]
        e_cho = g.edge_matrix[chosen[0], chosen[1]]
        # dL/dw > 0 on expert edges and < 0 on PP's edges, so a descent step
        # makes the expert route cheaper and PP's current route dearer
        assert r.grad[e_exp] > 0 > r.grad[e_cho]
        w_new = np.ones(g.num_edges) - 0.1 * r.grad
        assert planner.solve([s], [t], None, w_new)[0] == expert


def test_usage_loss_gradient():
    y, ys = np.array([2.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0])
    loss, grad = usage_loss(y, ys, "mean")
    assert np.isclose(loss, 2 / 3)
    assert np.allclose(grad, 2 * (y - ys) / 3)
