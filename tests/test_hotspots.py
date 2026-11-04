"""Collision hotspots on hand-made plans."""

from mapf_nco.grid import Grid
from mapf_nco.hotspots import close_calls, gave_way, naive_collisions
from mapf_nco.metrics import validate_solution

# A corridor: row 1, columns 1..5.
CORRIDOR = ["#######",
            "#.....#",
            "#######"]


def _grid(tmp_path):
    p = tmp_path / "corridor.map"
    p.write_text("type octile\nheight 3\nwidth 7\nmap\n" + "\n".join(CORRIDOR) + "\n")
    return Grid.from_file(str(p))


def test_gave_way_counts_waits(tmp_path):
    g = _grid(tmp_path)
    a, b, c = (g.node_of[(1, k)] for k in (1, 2, 3))
    cells, n = gave_way([[a, a, b, c], [c]])
    assert n == 1 and cells[a] == 1


def test_close_call_when_following_one_step_behind(tmp_path):
    g = _grid(tmp_path)
    v = [g.node_of[(1, k)] for k in range(1, 6)]
    leader = [v[1], v[2], v[3], v[4]]  # ends at column 5
    follower = [v[0], v[1], v[2], v[3]]  # enters each cell the leader just left
    validate_solution(g, [leader, follower], [v[1], v[0]], [v[4], v[3]])
    cells, n = close_calls([leader, follower])
    assert n == 3 and set(cells) == {v[1], v[2], v[3]}
    far = [v[0], v[0], v[1], v[2]]  # two steps behind: not a close call
    assert close_calls([leader, far])[1] == 0


def test_naive_collisions_head_on(tmp_path):
    g = _grid(tmp_path)
    left, right = g.node_of[(1, 1)], g.node_of[(1, 5)]
    cells, n = naive_collisions(g, [left, right], [right, left])
    assert n >= 1 and cells  # the two shortest paths meet in the corridor
    assert naive_collisions(g, [left], [right])[1] == 0  # one agent alone never collides


# ---------------------------------------------------------------- reference checks
def _pos(p, t):
    return p[t] if t < len(p) else p[-1]


def _brute_force(paths):
    """Deliberately simple pairwise versions of gave_way / close_calls / collisions."""
    from collections import Counter

    T = max(len(p) for p in paths)
    gw, cc, col = Counter(), Counter(), Counter()
    for p in paths:
        for t in range(len(p) - 1):
            if p[t] == p[t + 1]:
                gw[p[t]] += 1
    for i, a in enumerate(paths):
        for j, b in enumerate(paths):
            if i == j:
                continue
            for t in range(T - 1):
                if _pos(a, t) == _pos(b, t + 1):  # j enters where i was one step before
                    cc[_pos(b, t + 1)] += 1
    for t in range(T):
        here = sorted(_pos(p, t) for p in paths)
        for k in range(1, len(here)):
            if here[k] == here[k - 1]:
                col[here[k]] += 1  # every agent beyond the first in a cell
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            for t in range(1, T):
                u, v = _pos(paths[i], t - 1), _pos(paths[i], t)
                if u != v and _pos(paths[j], t - 1) == v and _pos(paths[j], t) == u:
                    col[u] += 1
                    col[v] += 1
    return gw, cc, col


def test_hotspots_match_brute_force():
    from mapf_nco.hotspots import individual_paths
    from mapf_nco.scenarios import assign_priority_order, make_rng, random_scenario
    from mapf_nco.solver_pp import PrioritizedPlanner

    g = Grid.from_file("data/random-32-32-20.map")
    pp = PrioritizedPlanner(g)
    for seed in range(12):
        sc = random_scenario(g, 25 + 5 * (seed % 4), make_rng(500 + seed))
        assert assign_priority_order(g, sc, make_rng(500 + seed))
        plan = pp.solve(sc.starts, sc.goals, sc.order)
        gw, cc, _ = _brute_force(plan)
        assert gave_way(plan)[0] == gw and close_calls(plan)[0] == cc
        assert sum(_brute_force(plan)[2].values()) == 0  # PP's plan itself never collides
        naive = individual_paths(g, sc.starts, sc.goals)
        assert naive_collisions(g, sc.starts, sc.goals)[0] == _brute_force(naive)[2]
