"""Sequential cost prediction (Gelato-style): recompute the costs before each agent.

The paper's model predicts ONE cost map per scenario before planning starts.
Here the model is called again before every agent's search (the priority order
stays fixed) and sees the current reservation state, so agent k plans on

    w_k = N_theta(map, starts/goals, reservations of agents 1..k-1, agent k)

This mirrors Gelato (graph edit distance via autoregressive neural combinatorial
optimisation, ICLR 2026), where each decision is conditioned on the partial
solution built so far. Two ways to train it:

* ``seq_blackbox``: run the whole sequential PP, then the paper's black-box
  gradient per agent: every agent k gets w_k' = w_k + lambda dL/dy, PP is rerun
  with those per-agent costs, and agent k's gradient is -(y_k - y_k')/lambda.
  (The dependence of w_k on earlier paths goes through discrete reservations
  and is not differentiated.)
* ``seq_teacher``: Gelato's "partial solutions" training. For every agent k the
  reservations come from the EXPERT's paths of agents 1..k-1, and agent k's step
  is trained (same loss and black-box gradient, on that single step) to produce
  the expert's path for agent k. EECBS's paths are mutually collision-free, so
  the target is always feasible; if every step is right, sequential PP
  reproduces the expert plan exactly.

Reservation features per cell (SEQ_DIM = 7):
    0 number of timesteps reserved by earlier agents          (/ 64)
    1 earliest reserved timestep                               (/ 64, 0 if none)
    2 latest reserved timestep                                 (/ 64)
    3 an earlier agent is parked here for good (its goal)
    4 progress k / n (same for every cell)
    5 the current agent's start
    6 the current agent's goal
"""

from __future__ import annotations

import numpy as np

from mapf_nco.blackbox import makespan_loss, makespan_perturbation, usage_loss
from mapf_nco.metrics import edge_usage
from mapf_nco.solver_pp import NoSolutionError, ReservationTable

SEQ_DIM = 7
T_SCALE = 64.0
METHODS = ("static", "seq_blackbox", "seq_teacher")


class ReservationFeatures:
    """Incrementally maintained per-cell features of the planned paths."""

    def __init__(self, grid):
        N = grid.num_nodes
        self.count = np.zeros(N, np.float32)
        self.first = np.full(N, np.inf, np.float32)
        self.last = np.zeros(N, np.float32)
        self.parked = np.zeros(N, np.float32)

    def add(self, path):
        p = np.asarray(path, dtype=np.int64)
        ts = np.arange(len(p), dtype=np.float32)
        np.add.at(self.count, p, 1.0)
        np.minimum.at(self.first, p, ts)
        np.maximum.at(self.last, p, ts)
        self.parked[p[-1]] = 1.0

    def current(self, start, goal, k, n):
        x = np.zeros((len(self.count), SEQ_DIM), np.float32)
        x[:, 0] = self.count / T_SCALE
        x[:, 1] = np.where(np.isfinite(self.first), self.first, 0.0) / T_SCALE
        x[:, 2] = self.last / T_SCALE
        x[:, 3] = self.parked
        x[:, 4] = k / max(n, 1)
        x[start, 5] = 1.0
        x[goal, 6] = 1.0
        return x


def _plan_one(planner, start, goal, w, res):
    wi = np.maximum(np.asarray(w, dtype=np.float64), planner.w_min)
    h = planner.grid.weighted_distances_to([goal], wi)[0]
    return planner._space_time_astar(start, goal, h, wi, res)


def sequential_pp(planner, starts, goals, order, cost_fn=None, fixed_weights=None,
                  offsets=None):
    """PP where agent k's costs come from ``cost_fn(features_k)``.

    With ``fixed_weights`` (one cost vector per agent) the model is not called
    (used for the perturbed pass). ``offsets``: optional extra cost per agent.
    Returns (paths, weights, features), each indexed by agent.
    """
    n = len(starts)
    res = ReservationTable(planner.grid.num_nodes)
    feats = ReservationFeatures(planner.grid)
    paths, W, X = [None] * n, [None] * n, [None] * n
    for k, i in enumerate(order):
        if fixed_weights is not None:
            w = fixed_weights[i]
        else:
            X[i] = feats.current(starts[i], goals[i], k, n)
            w = cost_fn(X[i])
        W[i] = w
        wi = w if offsets is None else w + offsets[i]
        path = _plan_one(planner, starts[i], goals[i], wi, res)
        if path is None:
            raise NoSolutionError(i)
        res.reserve(path)
        feats.add(path)
        paths[i] = path
    return paths, W, X


# ---------------------------------------------------------------- training signals
def seq_blackbox_signal(planner, starts, goals, order, cost_fn, y_star, lam, reduction,
                        objective="usage", tau=1.0, beta=1.0):
    """Per-agent black-box gradients through the whole sequential PP.

    Returns (loss, X, G, ok): features and dL/dw_k for every agent (None if PP failed).
    """
    grid = planner.grid
    try:
        paths, W, X = sequential_pp(planner, starts, goals, order, cost_fn)
    except NoSolutionError:
        return float("nan"), None, None, False
    y_k = [edge_usage(grid, [p]) for p in paths]
    y_hat = np.sum(y_k, axis=0)
    loss, dL_dy, offsets = 0.0, np.zeros(grid.num_edges), None
    if objective in ("usage", "usage+makespan"):
        loss, dL_dy = usage_loss(y_hat, y_star, reduction)
    if objective in ("makespan", "usage+makespan"):
        scale = beta if objective == "usage+makespan" else 1.0
        ms, alpha = makespan_loss(paths, tau)
        loss += scale * ms
        w_mean = np.mean(W, axis=0)
        offsets = [lam * scale * o for o in
                   makespan_perturbation(planner, starts, goals, w_mean, alpha)]
    W_pert = [np.maximum(w + lam * dL_dy, planner.w_min) for w in W]
    try:
        paths_p, _, _ = sequential_pp(planner, starts, goals, order, fixed_weights=W_pert,
                                      offsets=offsets)
    except NoSolutionError:
        return loss, X, [np.zeros(grid.num_edges)] * len(paths), True
    G = [-(y_k[i] - edge_usage(grid, [paths_p[i]])) / lam for i in range(len(paths))]
    return loss, X, G, True


def seq_teacher_signal(planner, starts, goals, order, cost_fn, expert_paths, lam, reduction):
    """Gelato-style teacher forcing on partial expert plans.

    Agent k sees the reservations of the EXPERT's paths of agents 1..k-1 and is
    trained (usage loss on its own path, black-box gradient on its single search
    step) to reproduce the expert's path. Returns (mean loss, X, G, ok, exact),
    where ``exact`` counts the steps whose path equals the expert's.
    """
    grid = planner.grid
    n = len(starts)
    res = ReservationTable(grid.num_nodes)
    feats = ReservationFeatures(grid)
    X, G, losses, exact = [None] * n, [None] * n, [], 0
    for k, i in enumerate(order):
        X[i] = feats.current(starts[i], goals[i], k, n)
        w = cost_fn(X[i])
        target = expert_paths[i]
        path = _plan_one(planner, starts[i], goals[i], w, res)
        G[i] = np.zeros(grid.num_edges)
        if path is not None:
            y_hat = edge_usage(grid, [path])
            y_star = edge_usage(grid, [target])
            loss, dL_dy = usage_loss(y_hat, y_star, reduction)
            losses.append(loss)
            exact += list(path) == list(target)
            p2 = _plan_one(planner, starts[i], goals[i], w + lam * dL_dy, res)
            if p2 is not None:
                G[i] = -(y_hat - edge_usage(grid, [p2])) / lam
        res.reserve(target)  # the next agents see the EXPERT's partial plan
        feats.add(target)
    return (float(np.mean(losses)) if losses else float("nan")), X, G, bool(losses), exact


# ------------------------------------------------------------------ worker side
_WORKER = {}


def model_spec(kind, model_kwargs, model):
    """Picklable description of a model (weights as NumPy arrays)."""
    def to_np(v):
        return v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)

    return (kind, dict(model_kwargs), {k: to_np(v) for k, v in model.state_dict().items()})


def _worker_model(spec, map_path):
    import torch

    from mapf_nco.grid import Grid
    from mapf_nco.models import build_model

    if _WORKER.get("map") != map_path:
        _WORKER.clear()
        _WORKER["map"] = map_path
        _WORKER["grid"] = Grid.from_file(map_path)
    kind, kwargs, state = spec
    key = (kind, tuple(sorted(kwargs.items())))
    if _WORKER.get("key") != key:
        _WORKER["model"] = build_model(kind, _WORKER["grid"], **kwargs)
        _WORKER["key"] = key
    model = _WORKER["model"]
    model.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in state.items()})
    model.eval()
    return model


def cost_fn_for(model, scenario_like):
    """Cost function x -> w (float64) for one scenario, as PP will use it."""
    import torch

    from mapf_nco.models import make_batch

    def fn(x):
        with torch.no_grad():
            return model(make_batch([scenario_like], seq=x[None]))[0].double().numpy()
    return fn


class _Sc:  # minimal scenario object for make_batch
    def __init__(self, starts, goals, order):
        self.starts, self.goals, self.order = starts, goals, order


def seq_train_task(args):
    """Worker: training signal of one scenario for a sequential method."""
    import torch

    from mapf_nco.solver_pp import PrioritizedPlanner

    (spec, map_path, w_min, starts, goals, order, expert_paths, y_star, method, lam,
     reduction, obj) = args
    torch.set_num_threads(1)
    model = _worker_model(spec, map_path)
    planner = PrioritizedPlanner(_WORKER["grid"], w_min=w_min)
    fn = cost_fn_for(model, _Sc(starts, goals, order))
    exact = 0
    if method == "seq_teacher":
        loss, X, G, ok, exact = seq_teacher_signal(planner, starts, goals, order, fn,
                                                  expert_paths, lam, reduction)
    else:
        loss, X, G, ok = seq_blackbox_signal(planner, starts, goals, order, fn, y_star, lam,
                                             reduction, **obj)
    if not ok:
        return {"loss": loss, "ok": False, "X": None, "G": None, "exact": 0}
    return {"loss": loss, "ok": True, "exact": exact,
            "X": np.stack(X).astype(np.float32), "G": np.stack(G).astype(np.float32)}


def seq_plan_task(args):
    """Worker: sequential PP with the model; returns (soc, makespan, seconds)."""
    import time

    import torch

    from mapf_nco.solver_pp import PrioritizedPlanner

    spec, map_path, w_min, starts, goals, order = args
    torch.set_num_threads(1)
    model = _worker_model(spec, map_path)
    planner = PrioritizedPlanner(_WORKER["grid"], w_min=w_min)
    t0 = time.perf_counter()
    try:
        paths, _, _ = sequential_pp(planner, starts, goals, order,
                                    cost_fn_for(model, _Sc(starts, goals, order)))
    except NoSolutionError:
        return -1, -1, time.perf_counter() - t0
    return (sum(len(p) - 1 for p in paths), max(len(p) - 1 for p in paths),
            time.perf_counter() - t0)
