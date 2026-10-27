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


def blackbox_step(planner: PrioritizedPlanner, starts, goals, order, w, y_star,
                  lam=1.0, reduction="mean"):
    """Forward pass, perturbed pass and surrogate gradient for one scenario."""
    w = np.asarray(w, dtype=np.float64)
    zero = np.zeros_like(w)
    t0 = time.perf_counter()
    try:
        paths = planner.solve(starts, goals, order, w)
    except NoSolutionError:
        return BlackBoxResult(float("nan"), zero, False, False, -1, time.perf_counter() - t0)
    pp_time = time.perf_counter() - t0
    y_hat = edge_usage(planner.grid, paths)
    loss, dL_dy = usage_loss(y_hat, y_star, reduction)

    w_pert = np.maximum(w + lam * dL_dy, planner.w_min)
    try:
        y_lam = edge_usage(planner.grid, planner.solve(starts, goals, order, w_pert))
    except NoSolutionError:
        return BlackBoxResult(loss, zero, True, False, sum_of_costs(paths), pp_time)
    grad = -(y_hat - y_lam) / lam
    return BlackBoxResult(loss, grad, True, True, sum_of_costs(paths), pp_time)
