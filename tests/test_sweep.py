from mapf_nco.sweep import apply_settings, expand_grid, second_agent_count


def test_expand_grid_is_cartesian_product():
    combos = expand_grid({"lr": [1e-4, 5e-4], "lam": [0.5, 1.0, 2.0], "epochs": [30]})
    assert len(combos) == 6
    assert {"lr": 1e-4, "lam": 2.0, "epochs": 30} in combos


def test_second_agent_count():
    cfg = {"agent_counts": [50, 75, 100]}
    assert second_agent_count(100, cfg) == 50
    assert second_agent_count(50, cfg) == 25
    assert second_agent_count(20, cfg) == 100  # below 50: test on the largest count


def test_apply_settings_overrides_only_given_keys():
    base = {"lr": 1.0, "lam": 2.0}
    cfg = apply_settings(base, {"lr": 0.5, "planner": "cbs"})
    assert cfg == {"lr": 0.5, "lam": 2.0, "planner": "cbs"} and base["lr"] == 1.0


def test_agent_scaling_and_targets():
    import pytest

    from conftest import MAP_PATH
    from mapf_nco.grid import Grid
    from mapf_nco.sweep import ALL_MAPS, resolve_map, scaled_agents, sweep_targets

    cfg = {"map_path": MAP_PATH}
    base = Grid.from_file(MAP_PATH)
    big = Grid.from_file(resolve_map(cfg, "random-64-64-20"))
    assert [scaled_agents(20, base, big, s) for s in ("count", "count_scale_2", "count_scale_4")] \
        == [20, 40, 80]
    assert scaled_agents(20, base, base, "count_scale_4") == 20  # the configured map itself
    with pytest.raises(ValueError):
        scaled_agents(20, base, big, "count_scale_3")
    with pytest.raises(FileNotFoundError):
        resolve_map(cfg, "random-128-128-5")
    t = sweep_targets(cfg, 20, list(ALL_MAPS), ["count", "count_scale_4"])
    assert [(x["grid"].name, x["agents"]) for x in t] == [
        ("random-32-32-20", 20), ("random-64-64-10", 20), ("random-64-64-10", 80),
        ("random-64-64-20", 20), ("random-64-64-20", 80)]
    assert t[0]["scalings"] == ["count", "count_scale_4"]  # the same run on the 32x32 map
    assert [(x["grid"].name, x["agents"]) for x in sweep_targets(cfg, 20)] == \
        [("random-32-32-20", 20)]  # default: the configured map only


def test_sweep_runs_on_every_map(tmp_path):
    import json
    import os

    import pytest

    pytest.importorskip("torch")
    from conftest import MAP_PATH
    from mapf_nco.sweep import run_sweep

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root_dir, "configs", "paper.json")) as f:
        cfg = json.load(f)
    cfg.update(map_path=MAP_PATH, data_dir=str(tmp_path / "data"),
               output_dir=str(tmp_path / "out"), expert="cbs", num_train=8, num_test=2,
               epochs=1, batch_size=4, planner="pp", model="edge", cbs_time_limit=10,
               time_limit=10, scp_dest=None)
    root, rows = run_sweep(cfg, 3, {"lr": [5e-4]}, test_count=1, workers=1,
                           log=lambda *a: None, maps=["random-32-32-20", "random-64-64-20"],
                           scalings=["count_scale_2"])
    assert [(r["map"], r["agents"]) for r in rows] == [("random-32-32-20", 3),
                                                       ("random-64-64-20", 6)]
    for r in rows:
        rd = os.path.join(root, "exp_01", r["map"], f"n{r['agents']}")
        assert os.path.exists(os.path.join(rd, "models", "model_edge.pt"))
        assert os.path.exists(os.path.join(rd, "test", "test_summary.md"))
        assert "test_orig_delay" in r and "test_edge_delay" in r
        assert os.path.exists(os.path.join(cfg["data_dir"], r["map"], f"n{r['agents']}",
                                           "train.json.gz"))  # generated for each map
    md = open(os.path.join(root, "sweep_summary.md"), encoding="utf-8").read()
    assert "| map | agents |" in md and "random-64-64-20 | 6 |" in md
    assert os.path.exists(os.path.join(root, "sweep_summary.png"))
