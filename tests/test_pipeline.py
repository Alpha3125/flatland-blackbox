"""Potentials for CBS: tie-breaking CBS, the delay search and the pipeline."""

import itertools
import os

import numpy as np
import pytest

from mapf_nco.grid import Grid
from mapf_nco.metrics import find_conflicts, sum_of_costs, validate_solution
from mapf_nco.scenarios import make_rng, random_scenario
from mapf_nco.solver_cbs import (TIE_MODES, ConflictTable, TieBreakCBS, _low_level_tb,
                                 solve_cbs)


def _path_cost(g, path, w):
    return sum(w[g.edge_matrix[u, v]] for u, v in zip(path[:-1], path[1:]))


# ------------------------------------------------------------ tie-breaking CBS
@pytest.mark.parametrize("seed", range(12))
def test_tiebreak_cbs_is_optimal_in_every_mode(seed):
    """Every tie-breaking mode returns the brute-force minimum sum of costs."""
    from test_correctness import _brute_force_min_soc

    g = Grid(["....", ".@..", "....", "..@."], name="tiny")
    cells = make_rng(seed).sample(g.largest_component(), 6)
    starts, goals = cells[:3], cells[3:]
    w = np.random.default_rng(seed).uniform(0.3, 3.0, g.num_edges)
    best = _brute_force_min_soc(g, starts, goals)
    for mode in TIE_MODES:
        paths = TieBreakCBS(g, mode, time_limit=30).solve(starts, goals, w)
        assert not find_conflicts(paths)
        assert sum_of_costs(paths) == best, mode


def test_tiebreak_modes_agree_on_the_real_map(grid):
    w = np.random.default_rng(1).uniform(0.5, 2.0, grid.num_edges)
    for seed in range(6):
        sc = random_scenario(grid, 10, make_rng(300 + seed))
        ref = solve_cbs(grid, sc.starts, sc.goals, time_limit=20)
        socs = set()
        for mode in TIE_MODES:
            cbs = TieBreakCBS(grid, mode, time_limit=20)
            paths = cbs.solve(sc.starts, sc.goals, w)
            validate_solution(grid, paths, sc.starts, sc.goals)
            socs.add(sum_of_costs(paths))
            assert cbs.last_nodes >= 1 and cbs.last_time > 0
        assert socs == {sum_of_costs(ref)}
    with pytest.raises(ValueError):
        TieBreakCBS(grid, "learned").solve(sc.starts, sc.goals)  # needs weights
    with pytest.raises(ValueError):
        TieBreakCBS(grid, "fastest")


def _shortest_paths(g, s, t):
    """Every shortest path s -> t (open grid, by brute force over move sequences)."""
    d = int(g.bfs_distances(t)[s])
    out = []
    for moves in itertools.product(range(4), repeat=d):
        path, v = [s], s
        for m in moves:
            nxt = [u for u, _ in g.out_edges[v] if u != v]
            if m >= len(nxt):
                break
            v = sorted(nxt)[m]
            path.append(v)
        if len(path) == d + 1 and path[-1] == t and len(set(path)) == len(path):
            out.append(path)
    return out


def test_learned_tiebreak_picks_the_cheapest_shortest_path(open_grid):
    g = open_grid
    s, t = g.node(0, 0), g.node(3, 3)
    h = g.bfs_distances(t)
    rng = np.random.default_rng(0)
    for _ in range(5):
        w = rng.uniform(0.2, 5.0, g.num_edges)
        hw = g.weighted_distances_to([t], w)[0]
        path, cost = _low_level_tb(g, s, t, set(), set(), 0, h, w, hw)
        assert cost == h[s] == len(path) - 1  # still a shortest path
        best = min(_path_cost(g, p, w) for p in _shortest_paths(g, s, t))
        assert _path_cost(g, path, w) == pytest.approx(best)


def _constrained_paths(g, s, t, L, vcons):
    """Every space-time path s -> t with exactly L actions (waits allowed) avoiding vcons."""
    out = []

    def go(path):
        if len(path) == L + 1:
            if path[-1] == t:
                out.append(list(path))
            return
        for u, _ in g.out_edges[path[-1]]:
            if (u, len(path)) not in vcons and g.bfs_distances(t)[u] <= L - len(path):
                go(path + [u])

    go([s])
    return out


def test_learned_tiebreak_is_lexicographic_under_constraints():
    """With constraints (waits and detours), the path is still the shortest one and,
    among those, the cheapest under the learned costs (by brute force)."""
    g = Grid(["....", "....", "....", "...."], name="open4")
    s, t = g.node(0, 0), g.node(2, 2)
    h = g.bfs_distances(t)
    for seed in range(80):
        rng = np.random.default_rng(seed)
        vcons = {(int(rng.integers(g.num_nodes)), int(rng.integers(1, 5)))
                 for _ in range(int(rng.integers(3, 9)))}
        vcons = {c for c in vcons if c[0] != t}  # no goal constraints (simpler brute force)
        if not vcons:
            continue
        w = rng.uniform(0.2, 5.0, g.num_edges)
        path, cost = _low_level_tb(g, s, t, vcons, set(), max(tt for _, tt in vcons), h, w,
                                   g.weighted_distances_to([t], w)[0])
        L = next(L for L in range(4, 10) if _constrained_paths(g, s, t, L, vcons))
        assert cost == L == len(path) - 1, seed
        assert all((v, k) not in vcons for k, v in enumerate(path))
        best = min(_path_cost(g, p, w) for p in _constrained_paths(g, s, t, L, vcons))
        assert _path_cost(g, path, w) == pytest.approx(best), seed


def test_learned_costs_never_make_the_path_longer(open_grid):
    g = open_grid
    s, t = g.node(0, 0), g.node(0, 2)  # straight line of length 2
    w = np.full(g.num_edges, 100.0)  # the straight moves are very dear ...
    for (r1, c1), (r2, c2) in [((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (1, 2)),
                               ((1, 2), (0, 2))]:
        w[g.edge_matrix[g.node(r1, c1), g.node(r2, c2)]] = 0.01  # ... a detour is cheap
    path, cost = _low_level_tb(g, s, t, set(), set(), 0, g.bfs_distances(t), w,
                               g.weighted_distances_to([t], w)[0])
    assert path == [s, g.node(0, 1), t] and cost == 2


def test_conflict_table_and_cat_avoids_other_agents(open_grid):
    g = open_grid
    c = g.node(1, 1)
    other = [c] * 6  # waits in the centre-ish cell, then rests there (its goal)
    cat = ConflictTable([other])
    assert cat.count(g.node(0, 1), c, 1) == 1 and cat.count(g.node(0, 1), c, 9) == 1  # rests
    swap = ConflictTable([[g.node(0, 1), g.node(0, 0)]])  # other agent: (0,1) -> (0,0) at t=1
    assert swap.count(g.node(0, 0), g.node(0, 1), 1) == 1  # we go the opposite way: swap
    assert ConflictTable([other], skip=0).count(g.node(0, 1), c, 1) == 0
    s, t = g.node(0, 0), g.node(2, 2)
    path, cost = _low_level_tb(g, s, t, set(), set(), 0, g.bfs_distances(t), cat=cat)
    assert cost == 4 and c not in path


# ------------------------------------------------------------------ delay search
def _tiny_runs(grid, n=8, k=24, seed=0):
    from mapf_nco.pipeline import make_scenarios

    cfg = {"map_path": grid.path}
    return make_scenarios(cfg, n, k, 1234 + seed, workers=1)


def test_delay_evaluator_matches_direct_pp(grid):
    from mapf_nco.collision import collision_costs, scenario_features
    from mapf_nco.collision import init_theta as c_theta
    from mapf_nco.delay_search import DelayEvaluator
    from mapf_nco.potential import init_theta, potential_costs
    from mapf_nco.solver_pp import PrioritizedPlanner

    scs = _tiny_runs(grid, k=6)
    ev = DelayEvaluator(grid.path, 1e-3, scs, grid, workers=1)
    pp = PrioritizedPlanner(grid, w_min=1e-3)
    th_p, th_c = [0.1, 0.2, 0.3, 0.0], [0.05, 0.05, 0.03, 0.02]
    wp = potential_costs(grid, init_theta(grid, {"near": .1, "toward": .2, "corridor": .3}))
    d_p, _ = ev.delays("potential", th_p, range(6))
    d_c, _ = ev.delays("collision", th_c, range(6))
    for i, s in enumerate(scs):
        lb = s.lower_bound(grid)
        assert d_p[i] == sum_of_costs(pp.solve(s.starts, s.goals, s.order, wp)) - lb
        wc = collision_costs(scenario_features(grid, s.starts, s.goals, s.order), c_theta(
            {"naive": .05, "gave_way": .05, "close_call": .03, "clean": .02}))
        assert d_c[i] == sum_of_costs(pp.solve(s.starts, s.goals, s.order, wc)) - lb
        assert np.allclose(ev.costs("collision", th_c, i), np.maximum(wc, 1e-3))
    assert np.array_equal(ev.plain()[:6], ev.delays("potential", [0, 0, 0, 0], range(6))[0])
    ev.close()


@pytest.mark.parametrize("kind", ["potential", "collision"])
def test_delay_search_never_worse_than_plain_on_validation(grid, kind):
    from mapf_nco.delay_search import DelayEvaluator, search

    scs = _tiny_runs(grid, n=12, k=30)
    ev = DelayEvaluator(grid.path, 1e-3, scs, grid, workers=1)
    r = search(kind, ev, list(range(20)), list(range(20, 30)), None, evals=14, seed=0,
               log=lambda *a: None)
    ev.close()
    lo, hi = r["box"]
    assert r["evaluations"] == len(r["history"]) <= 14
    assert r["train_best"] <= r["train_plain"] and r["val_chosen"] <= r["val_plain"] + 1e-12
    assert all(lo <= t <= hi for t in r["theta"])
    assert r["history"][0][0] == [0.0] * 4  # plain PP is the first setting tried
    with pytest.raises(ValueError):
        search("edge", ev, [0], [1], None)


# ---------------------------------------------------------------------- pipeline
def test_variant_list():
    from mapf_nco.pipeline import variant_list

    v = variant_list({}, ["potential", "collision"])
    assert [k for k, *_ in v] == ["plain", "cat", "potential_tb", "potential_tb_cat",
                                  "potential_weighted", "collision_tb", "collision_tb_cat",
                                  "collision_weighted"]
    assert [k for k, *_ in variant_list({"pipeline_variants": ["learned"]}, ["collision"])] == \
        ["plain", "collision_tb"]  # plain CBS is always the reference
    with pytest.raises(ValueError):
        variant_list({"pipeline_variants": ["fast"]}, ["potential"])


def test_pipeline_smoke(grid, tmp_path):
    from conftest import MAP_PATH
    from mapf_nco.pipeline import run_pipeline

    cfg = {"map_path": MAP_PATH, "output_dir": str(tmp_path / "out"), "w_min": 1e-3, "seed": 0,
           "pipeline_train": 10, "pipeline_val": 6, "pipeline_test": 5, "pipeline_cbs_test": 3,
           "pipeline_evals": 8, "pipeline_plots": 1, "pipeline_time_limit": 20,
           "potential_init": {"toward": 0.2, "corridor": 0.2, "lanes": 0.0},
           "collision_init": {"naive": 0.05}, "scp_dest": None}
    log = []
    root, s = run_pipeline(cfg, 8, workers=1, log=log.append)
    assert root.endswith(os.path.join("pipeline", "exp_1"))
    assert {"pipeline_summary.md", "pipeline_summary.png", "pipeline_config.json",
            "pipeline_scenarios.json.gz"} <= set(os.listdir(root))
    rd = os.path.join(root, grid.name, "n8")
    for folder, files in (("training", ["training_search.json", "training_search.png",
                                        "training_search_potential.csv",
                                        "training_search_collision.csv"]),
                          ("test", ["test_per_instance.csv", "test_instance_000_comparison.png"])):
        names = os.listdir(os.path.join(rd, folder))
        assert set(files) <= set(names) and all(f.startswith(folder) for f in names)
    md = open(os.path.join(root, "pipeline_summary.md"), encoding="utf-8").read()
    assert "DIFFERENT" not in md and "identical" in md  # tie-breaking never changes the delay
    for key in ("cat", "potential_tb", "collision_tb_cat"):
        assert all(d == 0 for d in s["cbs"][key]["delay_diff"])
    assert all(d >= 0 for d in s["cbs"]["collision_weighted"]["delay_diff"])
    assert s["cbs"]["plain"]["solved"] == 3
    assert run_pipeline(cfg, 8, workers=1, log=lambda *a: None)[0].endswith("exp_2")


def test_pipeline_command_line():
    import main

    args = main.parse_args(["pipeline", "--agents", "25", "--model", "collision", "--train", "50",
                            "--time_limit", "15", "--variants", "plain", "learned"])
    cfg = main.load_config(main.DEFAULT_CONFIG, None,
                           **{k: getattr(args, k, None) for k in main.HPARAMS})
    assert cfg["pipeline_models"] == ["collision"] and cfg["pipeline_train"] == 50
    assert cfg["pipeline_time_limit"] == 15 and cfg["pipeline_variants"] == ["plain", "learned"]
    assert cfg["pipeline_val"] == 100  # from the config file


def test_paired_node_comparison():
    from mapf_nco.pipeline import _fmt_paired, _paired

    ref = [{"solved": True, "nodes": n} for n in (10, 4, 8, 3)]
    run = [{"solved": True, "nodes": 5}, {"solved": True, "nodes": 4},
           {"solved": False, "nodes": 99}, {"solved": True, "nodes": 12}]
    d = _paired(run, ref, 4)
    assert (d["n"], d["fewer"], d["same"], d["more"]) == (3, 1, 1, 1)
    assert d["ratio"] == pytest.approx((0.5 * 1.0 * 4.0) ** (1 / 3))
    assert _fmt_paired(d) == "1 / 1 / 1, ×1.26" and _fmt_paired(_paired(run, ref, 0)) == "-"
