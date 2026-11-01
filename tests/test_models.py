import os

import pytest

torch = pytest.importorskip("torch")

from mapf_nco.models import EdgeModel, GNNModel, make_batch  # noqa: E402
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


def test_gnn_starts_at_unit_costs_and_backprops(grid):
    m = GNNModel(grid, hidden=16, num_freqs=3, id_dim=4, mlp_hidden=16)
    w = m(make_batch(_scenarios(grid)))
    assert w.shape == (3, grid.num_edges)
    assert torch.allclose(w, torch.ones_like(w), atol=1e-6)
    w.backward(torch.randn_like(w))
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    opt.step()
    w2 = m(make_batch(_scenarios(grid)))
    assert (w2 > 0).all() and not torch.allclose(w2[0], w2[1])  # instance-specific


def test_train_and_evaluate_smoke(grid, tmp_path):
    from mapf_nco.evaluate import evaluate, format_table
    from mapf_nco.scenarios import dataset_path, save_dataset
    from mapf_nco.solver_cbs import solve_cbs
    from mapf_nco.train import train

    from conftest import MAP_PATH

    cfg = {
        "map_path": MAP_PATH, "data_dir": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "out"), "val_fraction": 0.25, "epochs": 1,
        "batch_size": 4, "lr": 5e-4, "lam": 1.0, "loss_reduction": "mean", "w_min": 1e-3,
        "gnn": {"hidden": 16, "num_freqs": 3, "id_dim": 4, "mlp_hidden": 16}, "seed": 0,
    }
    scs = _scenarios(grid, n=6, k=10)
    for s in scs:
        s.expert_paths = solve_cbs(grid, s.starts, s.goals, time_limit=10)
    save_dataset(dataset_path(cfg["data_dir"], grid.name, 6, "train"), grid, scs[:8])
    save_dataset(dataset_path(cfg["data_dir"], grid.name, 6, "test"), grid, scs[8:])
    for kind in ("edge", "gnn"):
        ckpt = train(cfg, 6, kind, workers=1, log=lambda *a: None)
        assert os.path.exists(ckpt)
    rows = evaluate(cfg, [6], ("edge", "gnn"), log=lambda *a: None)
    assert "gnn_delay" in rows[0] and "edge_delay" in rows[0]
    assert "| 6 |" in format_table(rows)


def test_sequential_gnn_reads_reservations(grid):
    """seq_dim > 0: same output as static at start; after training it depends on the reservations."""
    import numpy as np

    from mapf_nco.sequential import SEQ_DIM

    scs = _scenarios(grid)
    m = GNNModel(grid, hidden=16, num_freqs=3, id_dim=4, mlp_hidden=16, seq_dim=SEQ_DIM)
    x0 = np.zeros((3, grid.num_nodes, SEQ_DIM), np.float32)
    x1 = np.random.default_rng(0).random((3, grid.num_nodes, SEQ_DIM)).astype(np.float32)
    w0 = m(make_batch(scs, seq=x0))
    assert torch.allclose(w0, torch.ones_like(w0), atol=1e-6)
    w = m(make_batch(scs, seq=x1))
    w.backward(torch.randn_like(w))
    torch.optim.Adam(m.parameters(), lr=1e-2).step()
    a, b = m(make_batch(scs, seq=x0)), m(make_batch(scs, seq=x1))
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("method", ["seq_teacher", "seq_blackbox"])
def test_sequential_training_smoke(grid, tmp_path, method):
    from mapf_nco.evaluate import test_saved
    from mapf_nco.scenarios import dataset_path, save_dataset
    from mapf_nco.solver_cbs import solve_cbs
    from mapf_nco.train import train

    from conftest import MAP_PATH

    cfg = {
        "map_path": MAP_PATH, "data_dir": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "out"), "val_fraction": 0.25, "epochs": 1,
        "batch_size": 4, "lr": 5e-3, "lam": 1.0, "loss_reduction": "mean", "w_min": 1e-3,
        "gnn": {"hidden": 16, "num_freqs": 3, "id_dim": 4, "mlp_hidden": 16}, "seed": 0,
        "method": method,
    }
    scs = _scenarios(grid, n=6, k=10)
    for s in scs:
        s.expert_paths = solve_cbs(grid, s.starts, s.goals, time_limit=10)
    save_dataset(dataset_path(cfg["data_dir"], grid.name, 6, "train"), grid, scs[:8])
    save_dataset(dataset_path(cfg["data_dir"], grid.name, 6, "test"), grid, scs[8:])
    for kind in ("edge", "gnn"):
        train(cfg, 6, kind, workers=1, log=lambda *a: None)
    r = test_saved(cfg, 6, None, ("edge", "gnn"), log=lambda *a: None)
    assert "gnn_delay" in r
    out = tmp_path / "out" / grid.name / "n6" / "test" / "instance_000"
    assert (out / "cost_evolution_gnn.png").exists()
