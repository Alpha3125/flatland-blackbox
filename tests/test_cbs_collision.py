import os

import numpy as np

from mapf_nco.scenarios import make_rng, random_scenario
from mapf_nco.solver_cbs import WeightedCBS, conflict_edges


def test_conflict_edges(open_grid):
    g = open_grid
    a, b, c = g.node(0, 0), g.node(0, 1), g.node(0, 2)
    # vertex conflict at b, t=1: agent 0 moves a->b, agent 1 moves c->b
    paths = [[a, b], [c, b]]
    assert conflict_edges(g, ("vertex", 1, 0, 1, b), paths) == [int(g.edge_matrix[a, b]),
                                                                int(g.edge_matrix[c, b])]
    # an agent waiting in place: its wait self-loop is the edge
    paths = [[b, b], [a, b]]
    assert int(g.edge_matrix[b, b]) in conflict_edges(g, ("vertex", 1, 0, 1, b), paths)
    # swap: the two opposite moves
    assert sorted(conflict_edges(g, ("edge", 1, 0, 1, (a, b)), [[a, b], [b, a]])) == sorted(
        [int(g.edge_matrix[a, b]), int(g.edge_matrix[b, a])])


def test_weighted_cbs_counts_conflicts(grid):
    from mapf_nco.solver_pp import NoSolutionError

    cbs = WeightedCBS(grid, time_limit=10)
    seen = 0
    for seed in range(8):
        sc = random_scenario(grid, 10, make_rng(seed))
        try:
            cbs.solve(sc.starts, sc.goals, None, np.ones(grid.num_edges))
        except NoSolutionError:  # also counted on a timeout, but there is no last node
            assert len(cbs.last_conflict_edges) == 2 * cbs.last_conflicts
            continue
        # every expanded node but the last (conflict-free) one resolved a conflict
        assert cbs.last_conflicts == cbs.last_nodes - 1
        assert len(cbs.last_conflict_edges) == 2 * cbs.last_conflicts
        assert all(0 <= e < grid.num_edges for e in cbs.last_conflict_edges)
        seen += cbs.last_conflicts
    assert seen > 0  # some of these scenarios have collisions


def test_step_task_raises_costs_where_conflicts_happen(grid):
    from conftest import MAP_PATH
    from mapf_nco import cbs_collision as cc

    cc._init_worker(MAP_PATH, 1e-3, 30)
    for seed in range(20):
        sc = random_scenario(grid, 10, make_rng(seed))
        r = cc._step_task((sc.starts, sc.goals, np.ones(grid.num_edges), 1.0))
        assert r["ok"] and r["grad"].shape == (grid.num_edges,)
        if r["conflicts"] and np.any(r["grad"]):
            # the gradient moves usage away from the forward plan: d/dw < 0 on edges only
            # the perturbed plan uses, > 0 on edges only the forward plan uses
            assert np.any(r["grad"] > 0) and np.any(r["grad"] < 0)
            return
    raise AssertionError("no scenario with conflicts and a changed perturbed plan")


def test_collision_sweep_runs(tmp_path):
    import json

    import pytest

    pytest.importorskip("torch")
    from conftest import MAP_PATH
    from mapf_nco.cbs_collision import run_collision_sweep

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root_dir, "configs", "paper.json")) as f:
        cfg = json.load(f)
    cfg.update(map_path=MAP_PATH, data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "o"),
               expert="cbs", num_train=8, num_test=3, epochs=1, batch_size=4,
               cbs_time_limit=20, expert_time_limit=20, scp_dest=None)
    root, rows = run_collision_sweep(cfg, 8, {"model": ["edge", "cell"]}, test_count=2,
                                     workers=1, log=lambda *a: None, maps=["random-32-32-20"])
    assert [r["model"] for r in rows] == ["edge", "cell"]
    for r in rows:
        assert r["test_scenarios"] == 2 and r["best_epoch"] in (0, 1)
        assert r["test_orig_timeouts"] == 0
        assert "test_learned_conflicts" in r and "test_orig_seconds" in r
    run = os.path.join(root, "exp_01_model-edge", "random-32-32-20", "n8")
    for f in ("models/model_edge.pt", "training/training_curves.png",
              "training/training_history.csv", "test/test_summary.md",
              "test/test_per_instance.csv", "test/test_summary.png"):
        assert os.path.exists(os.path.join(run, f)), f
    assert os.path.exists(os.path.join(root, "cbs_collision_summary.png"))
    md = open(os.path.join(root, "cbs_collision_summary.md"), encoding="utf-8").read()
    assert "conflicts: unit / learned" in md
