"""Sequential (Gelato-style) training and the cell heat-map model."""

import os

import numpy as np
import pytest

from mapf_nco.metrics import edge_usage
from mapf_nco.planners import training_runs
from mapf_nco.scenarios import Scenario, assign_priority_order, make_rng, random_scenario
from mapf_nco.sequential import expert_states, next_moves, scenario_steps, state_step
from mapf_nco.solver_cbs import solve_cbs
from mapf_nco.solver_pp import PrioritizedPlanner

OBJ = {"objective": "usage", "tau": 1.0, "beta": 1.0}


def _scenario(grid, n=8, seed=0):
    sc = random_scenario(grid, n, make_rng(seed))
    assign_priority_order(grid, sc, make_rng(seed))
    sc.expert_paths = solve_cbs(grid, sc.starts, sc.goals, time_limit=20)
    return sc


def test_expert_states(grid):
    sc = _scenario(grid)
    states = expert_states(grid, sc)
    T = max(len(p) for p in sc.expert_paths) - 1
    assert [t for t, *_ in states] == list(range(T))
    t0, pos, nxt, y0 = states[0]
    assert pos == list(sc.starts) and np.array_equal(y0, edge_usage(grid, sc.expert_paths))
    for t, pos, nxt, y in states:
        for i, p in enumerate(sc.expert_paths):
            assert pos[i] == p[min(t, len(p) - 1)] and nxt[i] == p[min(t + 1, len(p) - 1)]
    assert states[-1][3].sum() == sum(1 for p in sc.expert_paths if len(p) - 1 == T)


def test_no_deviation_when_the_expert_is_pp(grid):
    sc = _scenario(grid)
    pp = PrioritizedPlanner(grid)
    sc.expert_paths = pp.solve(sc.starts, sc.goals, sc.order)
    first = expert_states(grid, sc)[0]
    dev, r = state_step(pp, first, sc.goals, sc.order, np.ones(grid.num_edges), 1.0, "mean", OBJ)
    assert dev is False and r is None


def test_deviation_gives_the_blackbox_gradient(grid):
    """A state where PP's next moves differ: the update is the paper's gradient on the
    remaining plans (here with a strong perturbation so PP's plan changes)."""
    pp = PrioritizedPlanner(grid)
    for seed in range(20):
        sc = _scenario(grid, n=12, seed=seed)
        st = expert_states(grid, sc)[0]
        plan = pp.solve(sc.starts, sc.goals, sc.order)
        if next_moves(plan) != list(st[2]):
            break
    else:
        pytest.skip("PP agreed with CBS on the first move in every scenario tried")
    dev, r = state_step(pp, st, sc.goals, sc.order, np.ones(grid.num_edges), 50.0, "sum", OBJ)
    assert dev is True and r.ok
    y_hat = edge_usage(grid, plan)
    assert r.loss == pytest.approx(float(((y_hat - st[3]) ** 2).sum()))
    assert np.any(r.grad)
    res = scenario_steps(pp, grid, sc, np.ones(grid.num_edges), 50.0, "sum", OBJ)
    assert res.states == len(expert_states(grid, sc)) and 1 <= res.deviations <= res.states


def test_training_runs_sequential_is_pp_only():
    cfg = {"planner": "both", "model": "edge", "train_method": "sequential"}
    assert training_runs(cfg) == [("edge", "pp")]
    assert training_runs(dict(cfg, train_method="blackbox")) == [("edge", "pp"), ("edge", "cbs")]


def _torch():
    return pytest.importorskip("torch")


def test_cell_model_costs_and_gradient(grid):
    torch = _torch()
    from mapf_nco.models import CellModel, build_model, make_batch

    assert isinstance(build_model("cell", grid), CellModel)
    h = np.random.default_rng(0).uniform(-0.2, 0.5, grid.num_nodes).astype(np.float32)
    m = build_model("cell", grid, init=h)
    sc = Scenario(starts=[0], goals=[1], order=[0])
    w = m(make_batch([sc, sc]))
    assert w.shape == (2, grid.num_edges)
    assert np.allclose(w[0].detach().numpy(), 1 + h[grid.edge_dst], atol=1e-6)
    g = np.random.default_rng(1).standard_normal((2, grid.num_edges)).astype(np.float32)
    w.backward(torch.from_numpy(g))
    expected = np.zeros(grid.num_nodes)
    np.add.at(expected, grid.edge_dst, g.sum(0))
    assert np.allclose(m.heat.grad.numpy(), expected, atol=1e-4)


@pytest.mark.parametrize("kind,update", [("edge", "state"), ("cell", "state"),
                                         ("edge", "batch"), ("cell", "batch")])
def test_sequential_training_smoke(grid, tmp_path, kind, update):
    _torch()
    import json

    from mapf_nco.paths import training_dir
    from mapf_nco.train import train
    from test_models import _tiny_setup

    cfg = dict(_tiny_setup(grid, tmp_path), train_method="sequential", seq_update=update,
               loss_reduction="sum", lam=20.0)
    log = []
    ckpt = train(cfg, 6, "pp", workers=1, log=log.append, kind=kind)
    assert os.path.exists(ckpt) and os.path.basename(ckpt) == f"model_{kind}.pt"
    assert any("method sequential" in line for line in log)
    with open(os.path.join(training_dir(cfg, grid.name, 6), "training_history.json")) as f:
        h = json.load(f)[kind]["history"][-1]
    assert h["states"] > 0 and 0 <= h["deviating_states"] <= h["states"]
    with pytest.raises(ValueError):
        train(cfg, 6, "cbs", workers=1, log=lambda *a: None, kind=kind)


# ------------------------------------------------------------------- blockade mode
def _one_agent_state(g, first):
    """One agent from (0,0) to (2,2) whose expert goes ``first`` ('down' or 'right')."""
    s, t = g.node(0, 0), g.node(2, 2)
    path = ([s, g.node(1, 0), g.node(2, 0), g.node(2, 1), t] if first == "down" else
            [s, g.node(0, 1), g.node(0, 2), g.node(1, 2), t])
    sc = Scenario(starts=[s], goals=[t], order=[0], expert_paths=[path])
    return sc, expert_states(g, sc)[0]


@pytest.mark.parametrize("kind", ["edge", "cell"])
def test_blockade_makes_pp_follow_the_expert(open_grid, kind):
    from mapf_nco.sequential import Blockade

    g = open_grid
    pp = PrioritizedPlanner(g)
    plain_first = pp.solve([g.node(0, 0)], [g.node(2, 2)], [0])[0][1]
    first = "right" if plain_first == g.node(1, 0) else "down"  # the other equally short way
    sc, st = _one_agent_state(g, first)
    blk = Blockade(g, kind, factor=3.0, cap=20.0, retries=5)
    p = np.ones(g.num_edges) if kind == "edge" else np.zeros(g.num_nodes)
    r = blk.state(pp, st, sc.goals, sc.order, p)
    assert r["deviated"] and r["fixed"] and r["updates"] >= 1
    c = blk.costs(p).copy()
    pp_edge = g.edge_matrix[g.node(0, 0), plain_first]
    assert c[pp_edge] > 1 and c.max() <= 20.0 and c.min() >= 1.0
    assert pp.solve(sc.starts, sc.goals, sc.order, c)[0][1] == st[2][0]  # now follows
    again = blk.state(pp, st, sc.goals, sc.order, p)
    assert not again["deviated"] and again["updates"] == 0  # no further change needed
    blk.relax(p, 0.5)
    c2 = blk.costs(p)
    assert np.allclose(c2 - 1, (c - 1) * 0.5)


def test_blockade_cap_and_kinds(open_grid):
    from mapf_nco.sequential import Blockade

    g = open_grid
    blk = Blockade(g, "edge", factor=100.0, cap=20.0)
    p = np.ones(g.num_edges)
    e = g.edge_matrix[g.node(0, 0), g.node(0, 1)]
    e2 = g.edge_matrix[g.node(0, 0), g.node(1, 0)]
    blk.update(p, [e], [e2])
    assert p[e] == 20.0 and p[e2] == 1.0
    p[e2] = 9.0
    blk.update(p, [e2], [e])  # roles swapped: the expert's move gets cheaper again
    assert p[e] == pytest.approx(1.0) and p[e2] == 20.0
    blk.update(p, [e], [e])  # a move the expert also makes is never blocked
    assert p[e] == 1.0
    with pytest.raises(ValueError):
        Blockade(g, "potential")
    with pytest.raises(ValueError):
        Blockade(g, "edge", factor=1.0)


@pytest.mark.parametrize("kind", ["edge", "cell"])
def test_blockade_training_smoke(grid, tmp_path, kind):
    _torch()
    import json

    from mapf_nco.paths import training_dir
    from mapf_nco.train import train
    from test_models import _tiny_setup

    cfg = dict(_tiny_setup(grid, tmp_path), train_method="sequential", seq_mode="blockade",
               epochs=2)
    log = []
    ckpt = train(cfg, 6, "pp", workers=1, log=log.append, kind=kind)
    assert os.path.exists(ckpt) and any("blockade" in line for line in log)
    with open(os.path.join(training_dir(cfg, grid.name, 6), "training_history.json")) as f:
        h = json.load(f)[kind]["history"]
    assert all(k in h[-1] for k in ("fixed_states", "blockade_updates", "edges_at_cap"))
    assert h[-1]["max_cost"] <= 20.0 + 1e-6 and h[-1]["train_loss"] is not None
    # a model with strengths instead of per-edge costs is skipped, not crashed
    assert train(cfg, 6, "pp", workers=1, log=log.append, kind="collision") is None
    assert any("SKIPPED" in line and "blockade" in line for line in log)
