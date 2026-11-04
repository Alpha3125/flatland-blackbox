"""Learn the potential / collision strengths by searching directly on PP's delay.

The function minimised is PP's average delay on the training scenarios,

    J(theta) = mean over scenarios of  SoC(PP plan with costs w(theta)) - lower bound,

with w(theta) = 1 + F @ theta (clipped at w_min): F = the map features for the
potential model (mapf_nco/potential.py; only the four strengths, the one-way lanes
stay 0), F(scenario) = the collision features for the collision model
(mapf_nco/collision.py). No expert is needed: the lower bound is the sum of each
agent's shortest-path length.

J is piecewise constant in theta (PP's plan only changes at discrete points), so
there is no useful gradient; with four strengths a direct search is cheap instead
(PP takes milliseconds per scenario):

  1. plain PP (theta = 0), the starting strengths (potential_init / collision_init),
     and random settings inside the search box;
  2. a pattern search from the best one: try +-step on each strength, move on an
     improvement, halve the step when nothing improves; when it has converged and
     budget is left, it restarts from the best of a few new random settings;
  3. the final strengths: the best of {the 5 best on training, plain PP, the start}
     on the *validation* scenarios. Plain PP is a candidate, so the result is never
     worse than plain PP on validation.

If PP fails with some costs (no path for an agent under its priority order), that
scenario counts with plain PP's delay (the test falls back the same way) and the
failure is reported.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

KINDS = ("potential", "collision")
DEFAULT_BOX = {"potential": (-0.3, 0.6), "collision": (-0.2, 0.4)}


def strength_names(kind):
    if kind == "potential":
        from mapf_nco.potential import STRENGTHS
        return STRENGTHS
    if kind == "collision":
        from mapf_nco.collision import FEATURES
        return FEATURES
    raise ValueError(f"the delay search learns 'potential' or 'collision', not {kind!r}")


def start_strengths(kind, init):
    """The starting strengths (dict from the config) as a vector; lanes are not searched."""
    names = strength_names(kind)
    init = dict(init or {})
    unknown = set(init) - set(names) - ({"lanes"} if kind == "potential" else set())
    if unknown:
        raise ValueError(f"unknown {kind}_init key(s) {sorted(unknown)}; use {', '.join(names)}")
    return np.array([float(init.get(k, 0.0)) for k in names])


def as_dict(kind, theta):
    return {k: round(float(v), 6) for k, v in zip(strength_names(kind), theta)}


# ------------------------------------------------------------- worker processes
_S = {}


def _init_worker(map_path, w_min, scenarios):
    from mapf_nco.grid import Grid
    from mapf_nco.potential import feature_matrix
    from mapf_nco.solver_pp import PrioritizedPlanner

    grid = Grid.from_file(map_path)
    _S.update(grid=grid, w_min=w_min, scenarios=scenarios, collision={},
              pp=PrioritizedPlanner(grid, w_min=w_min),
              potential=feature_matrix(grid)[:, :4].astype(np.float64))


def _features(kind, i):
    if kind == "potential":
        return _S["potential"]
    cache = _S["collision"]
    if i not in cache:
        from mapf_nco.collision import scenario_features

        starts, goals, order, _ = _S["scenarios"][i]
        cache[i] = scenario_features(_S["grid"], starts, goals, order).astype(np.float64)
    return cache[i]


def costs(kind, i, theta):
    """Edge costs of scenario i (in a worker) under strengths ``theta``."""
    return np.maximum(1.0 + _features(kind, i) @ np.asarray(theta, np.float64), _S["w_min"])


def _delay_task(args):
    """Delays of PP on scenarios ``idxs`` under (kind, theta); None where PP fails."""
    from mapf_nco.solver_pp import NoSolutionError

    kind, theta, idxs = args
    out = []
    for i in idxs:
        starts, goals, order, lb = _S["scenarios"][i]
        w = None if not np.any(theta) else costs(kind, i, theta)
        try:
            paths = _S["pp"].solve(starts, goals, order, w)
        except NoSolutionError:
            out.append(None)
            continue
        out.append(sum(len(p) - 1 for p in paths) - lb)
    return out


class DelayEvaluator:
    """PP's delay on a fixed list of scenarios, in a process pool (or in-process).

    Make one evaluator for all the scenarios of a run (training, validation and
    test) and select them by index: the in-process copy is module state.
    """

    def __init__(self, map_path, w_min, scenarios, grid, workers=1):
        data = [(list(s.starts), list(s.goals), list(s.order), int(s.lower_bound(grid)))
                for s in scenarios]
        self.n, self.workers = len(data), max(1, int(workers))
        args = (map_path, float(w_min), data)
        _init_worker(*args)  # in-process copy (used when workers == 1, and for costs())
        self.pool = (ProcessPoolExecutor(self.workers, initializer=_init_worker, initargs=args)
                     if self.workers > 1 else None)
        self._plain = None

    def _map(self, kind, theta, idxs):
        k = max(1, -(-len(idxs) // (4 * self.workers)))
        tasks = [(kind, theta, idxs[j:j + k]) for j in range(0, len(idxs), k)]
        res = map(_delay_task, tasks) if self.pool is None else self.pool.map(_delay_task, tasks)
        return [x for r in res for x in r]

    def plain(self):
        """Plain PP's delay on every scenario (unit costs)."""
        if self._plain is None:
            d = self._map("potential", np.zeros(4), list(range(self.n)))
            if any(v is None for v in d):
                raise RuntimeError("plain PP failed on a scenario whose priority order "
                                   "was chosen so that it succeeds")
            self._plain = np.array(d, float)
        return self._plain

    def delays(self, kind, theta, idxs):
        """(PP delay per scenario in ``idxs``, number of PP failures); a failure
        counts with plain PP's delay."""
        idxs = list(idxs)
        d = self._map(kind, np.asarray(theta, np.float64), idxs)
        fails = sum(v is None for v in d)
        plain = self.plain() if fails else None
        return np.array([plain[i] if v is None else v for i, v in zip(idxs, d)], float), fails

    def costs(self, kind, theta, i):
        """Edge costs of scenario ``i`` under strengths ``theta`` (in this process)."""
        return costs(kind, i, theta)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


# ------------------------------------------------------------------------ search
def search(kind, evaluator, train_idx, val_idx, init, box=None, evals=150, seed=0, log=print):
    """Find the strengths with the lowest PP delay (see the module docstring).

    Returns a dict: theta (chosen on validation), train/val delays of plain PP, the
    start and the result, and the history of every setting tried.
    """
    names = strength_names(kind)
    K = len(names)
    lo, hi = box or DEFAULT_BOX[kind]
    rng = np.random.default_rng(seed)
    tried = {}  # rounded theta -> (train delay, PP failures)
    history = []  # (theta, train delay, best so far)

    def J(theta):
        theta = np.round(np.clip(theta, lo, hi), 4) + 0.0
        key = tuple(theta)
        if key not in tried:
            d, fails = evaluator.delays(kind, theta, train_idx)
            tried[key] = (float(d.mean()), fails)
            best = min(v[0] for v in tried.values())
            history.append((list(map(float, theta)), tried[key][0], best, fails))
        return tried[key][0]

    zero = np.zeros(K)
    start = np.clip(start_strengths(kind, init), lo, hi)
    J(zero)
    J(start)
    for _ in range(max(0, evals // 3 - 2)):
        J(rng.uniform(lo, hi, K))
    best = np.array(min(tried, key=lambda k: tried[k][0]))
    while len(tried) < evals:
        step, min_step = (hi - lo) / 8, (hi - lo) / 128
        while len(tried) < evals and step >= min_step:  # pattern search from ``best``
            improved = False
            for k in range(K):
                for sign in (1, -1):
                    if len(tried) >= evals:
                        break
                    cand = best.copy()
                    cand[k] += sign * step
                    if J(cand) < J(best) - 1e-12:
                        best, improved = np.round(np.clip(cand, lo, hi), 4) + 0.0, True
                        break
            if not improved:
                step /= 2
        if len(tried) >= evals:
            break
        # converged with budget left: restart from the best of a few new random settings
        fresh = [np.round(rng.uniform(lo, hi, K), 4) + 0.0
                 for _ in range(min(8, evals - len(tried)))]
        best = min(fresh, key=J)

    # final choice on validation: the 5 best on training, plain PP and the start
    ranked = sorted(tried, key=lambda k: (tried[k][0], float(np.abs(k).sum())))
    cands = list(dict.fromkeys(ranked[:5] + [tuple(np.round(zero, 4) + 0.0),
                                             tuple(np.round(start, 4) + 0.0)]))
    val = {}
    for c in cands:
        d, fails = evaluator.delays(kind, np.array(c), val_idx)
        val[c] = (float(d.mean()), fails)
    chosen = min(cands, key=lambda c: (val[c][0], tried[c][0], float(np.abs(c).sum())))
    z, s0 = tuple(np.round(zero, 4) + 0.0), tuple(np.round(start, 4) + 0.0)
    res = {
        "kind": kind, "names": list(names), "theta": list(map(float, chosen)),
        "box": [lo, hi], "evaluations": len(tried),
        "train_plain": tried[z][0], "train_start": tried[s0][0],
        "train_best": tried[ranked[0]][0], "train_chosen": tried[chosen][0],
        "val_plain": val[z][0], "val_start": val[s0][0], "val_chosen": val[chosen][0],
        "val_fails": val[chosen][1], "train_fails": tried[chosen][1],
        "candidates": [{"theta": list(map(float, c)), "train": tried[c][0], "val": val[c][0]}
                       for c in cands],
        "history": history,
    }
    log(f"  {kind}: {len(tried)} settings tried; train delay plain {res['train_plain']:.3f}, "
        f"start {res['train_start']:.3f}, best {res['train_best']:.3f}")
    log(f"  {kind}: chosen on validation: {_fmt(names, chosen)}  (validation delay "
        f"{res['val_chosen']:.3f} vs plain PP {res['val_plain']:.3f}, "
        f"{_pct(res['val_chosen'], res['val_plain'])})")
    return res


def _fmt(names, theta):
    return ", ".join(f"{n} {float(t):+.3f}" for n, t in zip(names, theta))


def _pct(x, ref):
    return "n/a" if not ref else f"{100 * (x - ref) / ref:+.1f}%"


def default_workers():
    return max(1, (os.cpu_count() or 2) - 1)
