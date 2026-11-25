"""Black-box differentiation through PP (paper Sec. 3.2, Algorithm 1).

PP maps edge costs w to a directed edge-usage vector y(w), which is piecewise
constant. Following Pogancic et al. (2019):

    L(y_hat, y*) = 1/M * sum_e (y_hat_e - y*_e)^2            (Eq. 4)
    w'           = w + lambda * dL/dy_hat                     (Eq. 5)
    y_lambda     = PP(w')
    dL/dw       ~= -(1/lambda) * (y_hat - y_lambda)           (Eq. 6)

Two PP calls per scenario, no changes to PP's internals. Everything here is
NumPy so it can run inside worker processes; the PyTorch side only receives
the gradient with respect to w.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from mapf_nco.metrics import edge_usage, sum_of_costs
from mapf_nco.solver_pp import NoSolutionError, PrioritizedPlanner


@dataclass
class BlackBoxResult:
    loss: float
    grad: np.ndarray  # dL/dw, shape (M,)
    ok: bool  # forward PP succeeded
    perturbed_ok: bool
    soc: int  # sum of costs of the forward plan (-1 if failed)
    pp_time: float


def usage_loss(y_hat, y_star, reduction="mean"):
    """MSE between directed usage vectors and its gradient w.r.t. y_hat."""
    diff = y_hat - y_star
    if reduction == "mean":
        M = diff.shape[0]
        return float(diff @ diff / M), 2.0 * diff / M
    if reduction == "sum":
        return float(diff @ diff), 2.0 * diff
    raise ValueError(reduction)


def makespan_loss(paths, tau=1.0):
    """Smooth makespan: tau * log sum_i exp(T_i / tau), with T_i = sum_e y_{i,e}.

    Returns the loss and its gradient w.r.t. each agent's usage vector, which is
    the same for every edge of agent i: softmax_i(T / tau). The agents that
    finish last get (almost) all of it; tau -> 0 gives the exact maximum.
    """
    T = np.array([len(p) - 1 for p in paths], dtype=np.float64)
    z = T / tau
    m = z.max()
    lse = m + np.log(np.exp(z - m).sum())
    alpha = np.exp(z - lse)
    return float(tau * lse), alpha


def makespan_perturbation(planner, starts, goals, w, alpha, min_alpha=1e-3):
    """Per-agent cost changes that the makespan loss asks for.

    In PP each agent already takes its fastest route given the agents planned
    before it, so a late agent is late because others are in its way. The
    surrogate gradient of the smooth makespan with respect to agent j's usage is
    therefore taken as  sum_{i != j} alpha_i * r_i  (+ alpha_j on every step of
    agent j itself), where alpha_i = softmax_i(T / tau) is agent i's lateness
    weight and r_i marks every edge entering a cell of agent i's ideal route
    (its best route under w with nobody else around). Perturbing with it makes
    the other agents avoid the routes the late agents need.
    """
    grid = planner.grid
    late = [i for i in range(len(starts)) if alpha[i] >= min_alpha]
    route_cost = np.zeros(grid.num_edges)
    own = {}
    for i in late:
        (ideal,) = planner.solve([starts[i]], [goals[i]], None, w)
        r_i = np.isin(grid.edge_dst, np.unique(ideal)).astype(float)
        own[i] = alpha[i] * r_i
        route_cost += own[i]
    return [route_cost - own.get(j, 0.0) + alpha[j] for j in range(len(starts))]


def blackbox_step(planner: PrioritizedPlanner, starts, goals, order, w, y_star,
                  lam=1.0, reduction="mean", objective="usage", tau=1.0, beta=1.0):
    """Forward pass, perturbed pass and surrogate gradient for one scenario.

    objective:
      "usage"          paper: MSE between PP's and the expert's edge usage (Eq. 4)
      "makespan"       smooth max of the agents' arrival times (no expert needed)
      "usage+makespan" usage loss + beta * makespan loss

    The plan is a usage vector per agent (y_i); the shared costs w are given to
    every agent, so dL/dw is the sum of the per-agent Pogancic gradients, i.e.
    -(y_hat - y_lam)/lambda on the total usage. The usage loss perturbs every
    agent's costs equally (Eq. 5). For the makespan loss see
    :func:`makespan_perturbation`.
    """
    w = np.asarray(w, dtype=np.float64)
    zero = np.zeros_like(w)
    t0 = time.perf_counter()
    try:
        paths = planner.solve(starts, goals, order, w)
    except NoSolutionError:
        return BlackBoxResult(float("nan"), zero, False, False, -1, time.perf_counter() - t0)
    pp_time = time.perf_counter() - t0
    y_hat = edge_usage(planner.grid, paths)
    loss, dL_dy, offsets = 0.0, np.zeros_like(w), None
    if objective in ("usage", "usage+makespan"):
        loss, dL_dy = usage_loss(y_hat, y_star, reduction)
    if objective in ("makespan", "usage+makespan"):
        scale = beta if objective == "usage+makespan" else 1.0
        ms, alpha = makespan_loss(paths, tau)
        loss += scale * ms
        offsets = [lam * scale * o for o in makespan_perturbation(planner, starts, goals, w, alpha)]
    elif objective != "usage":
        raise ValueError(f"unknown objective {objective}")

    w_pert = np.maximum(w + lam * dL_dy, planner.w_min)
    try:
        y_lam = edge_usage(planner.grid, planner.solve(starts, goals, order, w_pert, offsets))
    except NoSolutionError:
        return BlackBoxResult(loss, zero, True, False, sum_of_costs(paths), pp_time)
    grad = -(y_hat - y_lam) / lam
    return BlackBoxResult(loss, grad, True, True, sum_of_costs(paths), pp_time)
