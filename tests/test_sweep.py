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
