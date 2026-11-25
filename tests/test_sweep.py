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


def test_fixed_positions_data(tmp_path):
    """positions=fixed: one scenario for every train and test instance, in its own folder."""
    import os

    import pytest

    from conftest import MAP_PATH
    from mapf_nco.experts import generate_datasets
    from mapf_nco.grid import Grid
    from mapf_nco.scenarios import data_dir_for, dataset_path, load_dataset

    grid = Grid.from_file(MAP_PATH)
    cfg = {"map_path": MAP_PATH, "data_dir": str(tmp_path / "d"), "expert": "cbs",
           "expert_time_limit": 10, "cbs_time_limit": 10, "num_train": 5, "num_test": 3,
           "seed": 0, "positions": "fixed"}
    assert data_dir_for(cfg) == os.path.join(cfg["data_dir"], "fixed_positions")
    generate_datasets(cfg, [4], workers=1, log=lambda *a: None)
    train, info = load_dataset(dataset_path(data_dir_for(cfg), grid.name, 4, "train"), grid)
    test, _ = load_dataset(dataset_path(data_dir_for(cfg), grid.name, 4, "test"), grid)
    assert len(train) == 5 and len(test) == 3 and info["positions"] == "fixed"
    first = train[0].to_dict()
    for sc in train + test:
        d = sc.to_dict()
        for k in ("starts", "goals", "order", "expert_paths", "pp_paths", "cbs_paths"):
            assert d[k] == first[k]
        assert sc.meta["fixed_positions"]
    assert not os.path.exists(os.path.join(cfg["data_dir"], grid.name))  # randomized untouched

    # a missing split is rebuilt from the saved one, so train and test stay the same scenario
    os.remove(dataset_path(data_dir_for(cfg), grid.name, 4, "test"))
    generate_datasets(dict(cfg, num_test=2), [4], workers=1, log=lambda *a: None)
    test, _ = load_dataset(dataset_path(data_dir_for(cfg), grid.name, 4, "test"), grid)
    assert len(test) == 2 and test[0].starts == train[0].starts

    # missing plans are added to the fixed scenario once, and the copies stay identical
    from mapf_nco.scenarios import save_dataset

    path = dataset_path(data_dir_for(cfg), grid.name, 4, "train")
    for sc in train:
        sc.pp_paths = None
    save_dataset(path, grid, train, info)
    generate_datasets(dict(cfg, augment=True), [4], workers=1, log=lambda *a: None)
    again, info2 = load_dataset(path, grid)
    assert len(again) == 5 and info2["positions"] == "fixed"
    assert all(s.to_dict()["pp_paths"] == first["pp_paths"] for s in again)

    rnd = dict(cfg, positions="randomized")
    generate_datasets(rnd, [4], workers=1, log=lambda *a: None)
    train_r, _ = load_dataset(dataset_path(data_dir_for(rnd), grid.name, 4, "train"), grid)
    assert len({tuple(s.starts) for s in train_r}) == 5
    with pytest.raises(ValueError):
        data_dir_for(dict(cfg, positions="random"))


def test_sweep_with_fixed_positions(tmp_path):
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
               output_dir=str(tmp_path / "out"), expert="cbs", num_train=8, num_test=4,
               epochs=1, batch_size=4, planner="pp", model="edge", cbs_time_limit=10,
               time_limit=10, scp_dest=None, positions="fixed")
    root, rows = run_sweep(cfg, 3, {"lr": [5e-4]}, test_count=3, second_n=4, workers=1,
                           log=lambda *a: None, maps=["random-32-32-20"])
    (r,) = rows
    assert r["positions"] == "fixed" and r["second_agents"] == 4
    assert os.path.exists(os.path.join(cfg["data_dir"], "fixed_positions", r["map"], "n3",
                                       "train.json.gz"))
    assert not os.path.exists(os.path.join(cfg["data_dir"], r["map"]))
    per = os.path.join(root, "exp_01", r["map"], "n3", "test", "test_per_instance.csv")
    assert len(open(per).read().strip().splitlines()) == 2  # header + the one instance
    md = open(os.path.join(root, "sweep_summary.md"), encoding="utf-8").read()
    assert "Start/goal positions: **fixed**" in md
