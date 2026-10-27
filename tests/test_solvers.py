import os
import stat
import sys
import textwrap

import pytest

from mapf_nco.metrics import delay, validate_solution
from mapf_nco.scenarios import (
    Scenario, assign_priority_order, load_dataset, make_rng, random_scenario, read_scen,
    save_dataset, write_scen,
)
from mapf_nco.solvers.cbs import solve_cbs
from mapf_nco.solvers.eecbs import parse_eecbs_paths, solve_eecbs
from mapf_nco.solvers.pp import solve_pp


def test_cbs_is_optimal_and_feasible(grid):
    for seed in range(3):
        sc = random_scenario(grid, 8, make_rng(seed))
        assign_priority_order(grid, sc, make_rng(seed))
        cbs = solve_cbs(grid, sc.starts, sc.goals, time_limit=30)
        assert cbs is not None
        validate_solution(grid, cbs, sc.starts, sc.goals)
        pp = solve_pp(grid, sc.starts, sc.goals, sc.order)
        assert delay(grid, cbs, sc.starts, sc.goals) <= delay(grid, pp, sc.starts, sc.goals)


def test_scen_roundtrip(grid, tmp_path):
    sc = random_scenario(grid, 10, make_rng(1))
    p = tmp_path / "x.scen"
    write_scen(str(p), grid, sc.starts, sc.goals)
    back = read_scen(str(p), grid)
    assert back.starts == sc.starts and back.goals == sc.goals
    # MovingAI convention: column (x) before row (y)
    first = p.read_text().splitlines()[1].split("\t")
    r, c = grid.cell(sc.starts[0])
    assert (int(first[4]), int(first[5])) == (c, r)


def test_dataset_roundtrip(grid, tmp_path):
    sc = random_scenario(grid, 5, make_rng(2))
    assign_priority_order(grid, sc, make_rng(2))
    sc.expert_paths = solve_pp(grid, sc.starts, sc.goals, sc.order)
    path = tmp_path / "d.json.gz"
    save_dataset(str(path), grid, [sc], {"k": 1})
    (back,), info = load_dataset(str(path), grid)
    assert back.to_dict()["expert_paths"] == sc.to_dict()["expert_paths"]
    assert back.order == sc.order and info == {"k": 1}


def test_parse_eecbs_output(open_grid):
    g = open_grid
    text = "Agent 0: (0,0)->(0,1)->(1,1)->\nAgent 1: (4,4)->(4,4)->\n"
    paths = parse_eecbs_paths(text, g)
    assert paths == [[g.node(0, 0), g.node(0, 1), g.node(1, 1)], [g.node(4, 4)] * 2]


FAKE_EECBS = """\
#!{python}
# Stand-in for the EECBS binary: same CLI and file formats, solves with PP.
import sys
sys.path.insert(0, {root!r})
from mapf_nco.grid import Grid
from mapf_nco.scenarios import read_scen
from mapf_nco.solvers.pp import solve_pp
a = sys.argv[1:]
opt = {{a[i]: a[i + 1] for i in range(0, len(a) - 1, 2)}}
g = Grid.from_file(opt["-m"])
sc = read_scen(opt["-a"], g, int(opt["-k"]))
paths = solve_pp(g, sc.starts, sc.goals)
with open(opt["--outputPaths"], "w") as f:
    for i, p in enumerate(paths):
        f.write("Agent %d: " % i + "".join("(%d,%d)->" % g.cell(v) for v in p) + "\\n")
"""


@pytest.mark.skipif(sys.platform == "win32", reason="shebang script")
def test_eecbs_wrapper_with_fake_binary(grid, tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fake = tmp_path / "eecbs"
    fake.write_text(FAKE_EECBS.format(python=sys.executable, root=root))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    sc = random_scenario(grid, 15, make_rng(5))
    paths = solve_eecbs(grid, sc.starts, sc.goals, str(fake), suboptimality=1.0, time_limit=10)
    assert paths is not None
    validate_solution(grid, paths, sc.starts, sc.goals)


def test_compare_writes_pictures(grid, tmp_path):
    from conftest import MAP_PATH
    from mapf_nco.compare import compare
    from mapf_nco.scenarios import dataset_path

    scs = []
    for seed in range(3):
        sc = random_scenario(grid, 6, make_rng(seed))
        assign_priority_order(grid, sc, make_rng(seed))
        sc.expert_paths = solve_cbs(grid, sc.starts, sc.goals, time_limit=10)
        scs.append(sc)
    cfg = {"map_path": MAP_PATH, "data_dir": str(tmp_path / "d"), "output_dir": str(tmp_path / "o"),
           "w_min": 1e-3, "expert": "cbs", "expert_time_limit": 10}
    save_dataset(dataset_path(cfg["data_dir"], grid.name, 6, "test"), grid, scs)
    quiet = lambda *a: None  # noqa: E731
    plans = compare(cfg, 6, "test", None, None, kinds=(), animate=True, log=quiet)
    assert "PP (orig)" in plans and "Expert" in plans
    compare(cfg, 6, seed=7, kinds=(), log=quiet)  # fresh random scenario
    out = tmp_path / "o" / grid.name / "n6" / "plots"
    assert (out / "n6_seed7" / "comparison.png").exists()
    assert any(out.glob("n6_test*/animation.gif"))


def test_wsl_path_translation(monkeypatch):
    import mapf_nco.solvers.eecbs as e

    monkeypatch.setattr(e, "IS_WINDOWS", True)
    monkeypatch.setattr(e.os.path, "abspath", lambda p: p)
    assert e.to_wsl_path(r"C:\Users\A\repo\data\x.map") == "/mnt/c/Users/A/repo/data/x.map"
    assert e.to_wsl_path(r"\\wsl$\Ubuntu\home\me\eecbs") == "/home/me/eecbs"
    assert e.uses_wsl("wsl:/home/me/eecbs")
