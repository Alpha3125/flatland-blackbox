import os

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from mapf_nco.models import EdgeModel, PotentialModel, build_model, make_batch  # noqa: E402
from mapf_nco.scenarios import assign_priority_order, make_rng, random_scenario  # noqa: E402


def _scenarios(grid, n=10, k=3):
    out = []
    for seed in range(k):
        sc = random_scenario(grid, n, make_rng(seed))
        assign_priority_order(grid, sc, make_rng(seed))
        out.append(sc)
    return out


def test_edge_model(grid):
    m = EdgeModel(grid.num_edges)
    w = m(make_batch(_scenarios(grid)))
    assert w.shape == (3, grid.num_edges)
    assert torch.allclose(w, torch.ones_like(w))
    w.backward(torch.ones_like(w))
    assert torch.allclose(m.edge_weights.grad, torch.full((grid.num_edges,), 3.0))


def test_models(grid):
    assert isinstance(build_model("edge", grid), EdgeModel)
    with pytest.raises(ValueError):
        build_model("gnn", grid)
    e = build_model("potential_edge", grid, init=np.full(grid.num_edges, 2.0))
    assert torch.allclose(e(make_batch(_scenarios(grid))), torch.full((3, grid.num_edges), 2.0))


def test_potential_model_costs_and_gradient(grid):
    from mapf_nco.potential import feature_matrix, init_theta, potential_costs

    init = {"toward": 0.2, "corridor": 0.3, "lanes": 0.1}
    m = build_model("potential", grid, init=init)
    assert isinstance(m, PotentialModel)
    w = m(make_batch(_scenarios(grid)))
    assert w.shape == (3, grid.num_edges)
    assert np.allclose(w[0].detach().numpy(), potential_costs(grid, init_theta(grid, init)),
                       atol=1e-5)
    g = np.random.default_rng(0).standard_normal((3, grid.num_edges)).astype(np.float32)
    w.backward(torch.from_numpy(g))
    expected = feature_matrix(grid).T @ g.sum(0)
    assert np.allclose(m.theta.grad.numpy(), expected, atol=1e-3)
    assert "features" not in m.state_dict()  # recomputed from the map, not stored


def test_collision_model_costs_and_gradient(grid):
    from mapf_nco.collision import collision_costs, init_theta, scenario_features

    scs = _scenarios(grid, n=20)
    init = {"naive": 0.05, "gave_way": 0.05, "close_call": 0.03, "clean": 0.02}
    m = build_model("collision", grid, init=init)
    w = m(make_batch(scs))
    Fs = [scenario_features(grid, s.starts, s.goals, s.order) for s in scs]
    for b, F in enumerate(Fs):  # costs differ per scenario
        assert np.allclose(w[b].detach().numpy(), collision_costs(F, init_theta(init)), atol=1e-5)
    g = np.random.default_rng(0).standard_normal((3, grid.num_edges)).astype(np.float32)
    w.backward(torch.from_numpy(g))
    expected = sum(F.T @ g[b] for b, F in enumerate(Fs))
    assert np.allclose(m.theta.grad.numpy(), expected, atol=1e-2)
    assert list(m.state_dict()) == ["theta"]


@pytest.mark.parametrize("planner,kinds", [(p, k) for p in ("pp", "cbs")
                                           for k in (("edge",), ("potential", "potential_edge"),
                                                     ("collision",))])
def test_train_and_evaluate_smoke(grid, tmp_path, planner, kinds):
    from mapf_nco.evaluate import evaluate, format_table
    from mapf_nco.planners import model_id
    from mapf_nco.scenarios import dataset_path, save_dataset
    from mapf_nco.solver_cbs import solve_cbs
    from mapf_nco.train import train

    from conftest import MAP_PATH

    cfg = {
        "map_path": MAP_PATH, "data_dir": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "out"), "val_fraction": 0.25, "epochs": 1,
        "batch_size": 4, "lr": 5e-4, "lam": 1.0, "loss_reduction": "mean", "w_min": 1e-3,
        "seed": 0, "cbs_train_time_limit": 10, "cbs_time_limit": 10,
        "potential_init": {"toward": 0.2, "corridor": 0.2},
        "collision_init": {"naive": 0.05, "clean": 0.02},
    }
    scs = _scenarios(grid, n=6, k=10)
    for s in scs:
        s.expert_paths = solve_cbs(grid, s.starts, s.goals, time_limit=10)
    save_dataset(dataset_path(cfg["data_dir"], grid.name, 6, "train"), grid, scs[:8])
    save_dataset(dataset_path(cfg["data_dir"], grid.name, 6, "test"), grid, scs[8:])
    for kind in kinds:
        ckpt = train(cfg, 6, planner, workers=1, log=lambda *a: None, kind=kind)
        assert os.path.exists(ckpt) and os.path.basename(ckpt) == f"{model_id(planner, kind)}.pt"
    rows = evaluate(cfg, [6], log=lambda *a: None)
    for kind in kinds:
        assert f"{model_id(planner, kind)}_delay" in rows[0]
    assert "| 6 |" in format_table(rows)
