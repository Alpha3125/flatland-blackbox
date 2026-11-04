"""Potential-based edge costs (PotP): features of the map, not of the scenario.

The cost of a directed move u -> v is

    w_e = 1 + theta_near     * near(v)          v is close to an obstruction
            + theta_toward   * toward(u, v)     the move gets closer to an obstruction
            + theta_corridor * corridor(u, v)   v is a single-lane passage in the move's direction
            + theta_dead_end * dead_end(v)      v is a dead end
            + lane_row[r]    * dc               horizontal move in row r (dc = +1 east, -1 west)
            + lane_col[c]    * dr               vertical move in column c (dr = +1 south, -1 north)

and waits keep cost 1. ``near(v) = 1 / d(v)`` is a repulsive potential, with
d(v) the number of steps from v to the nearest blocked cell or the map border
(1 next to an obstacle). ``toward = max(0, near(v) - near(u))``. A cell is a
single-lane passage for a move if both cells beside it, perpendicular to the
move, are blocked (agents cannot pass each other there). A lane parameter
makes one direction along its row / column cheaper and the opposite one
dearer, like a one-way street; it is learned per row and per column.

Everything is linear in the parameters: w = 1 + F @ theta, with F a fixed
(M x P) matrix, P = 4 + height + width.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

STRENGTHS = ("near", "toward", "corridor", "dead_end")
DEFAULT_INIT = {"near": 0.0, "toward": 0.0, "corridor": 0.0, "dead_end": 0.0, "lanes": 0.0}


def obstacle_distance(grid):
    """Steps (4-neighbour) from every free cell to the nearest blocked cell or the border."""
    padded = np.pad(grid.free.astype(bool), 1, constant_values=False)
    d = ndimage.distance_transform_cdt(padded, metric="taxicab")[1:-1, 1:-1]
    r, c = grid.coords[:, 0], grid.coords[:, 1]
    return d[r, c].astype(np.float64)


def _blocked(grid, r, c):
    return not (0 <= r < grid.height and 0 <= c < grid.width) or not grid.free[r, c]


def feature_matrix(grid):
    """(M x P) float32 matrix F such that w = 1 + F @ theta; see the module docstring.

    Columns: the four STRENGTHS, then one lane column per row, then one per column.
    """
    H, W, M = grid.height, grid.width, grid.num_edges
    near = 1.0 / obstacle_distance(grid)
    F = np.zeros((M, len(STRENGTHS) + H + W), np.float32)
    for e in grid.move_edges:
        u, v = int(grid.edge_src[e]), int(grid.edge_dst[e])
        (ru, cu), (rv, cv) = grid.coords[u], grid.coords[v]
        dr, dc = int(rv - ru), int(cv - cu)
        F[e, 0] = near[v]
        F[e, 1] = max(0.0, near[v] - near[u])
        # cells beside v, perpendicular to the move direction (dr, dc)
        F[e, 2] = float(_blocked(grid, rv + dc, cv + dr) and _blocked(grid, rv - dc, cv - dr))
        F[e, 3] = float(grid.degree[v] == 1)
        if dr == 0:
            F[e, len(STRENGTHS) + ru] = dc  # horizontal move: lane of row ru
        else:
            F[e, len(STRENGTHS) + H + cu] = dr  # vertical move: lane of column cu
    return F


def init_theta(grid, init=None):
    """Starting parameters: the four strengths, and the lanes as an alternating
    one-way pattern of strength ``init['lanes']`` (even rows east, odd rows west;
    even columns south, odd columns north). All zeros = the original unit costs."""
    unknown = set(init or {}) - set(DEFAULT_INIT)
    if unknown:
        raise ValueError(f"unknown potential_init key(s) {sorted(unknown)}; "
                         f"use {', '.join(DEFAULT_INIT)}")
    init = {**DEFAULT_INIT, **(init or {})}
    lanes = float(init["lanes"])
    rows = np.where(np.arange(grid.height) % 2 == 0, -lanes, lanes)  # -: east is cheaper
    cols = np.where(np.arange(grid.width) % 2 == 0, -lanes, lanes)  # -: south is cheaper
    return np.concatenate([[float(init[k]) for k in STRENGTHS], rows, cols]).astype(np.float32)


def potential_costs(grid, theta, F=None):
    F = feature_matrix(grid) if F is None else F
    return 1.0 + F.astype(np.float64) @ np.asarray(theta, np.float64)


def describe(grid, theta):
    """Short text of the strengths and lane summary, for logs and plot headers."""
    theta = np.asarray(theta, float)
    k = len(STRENGTHS)
    s = ", ".join(f"{n} {theta[i]:+.3f}" for i, n in enumerate(STRENGTHS))
    lanes = np.abs(theta[k:])
    return s + f", lanes |mean| {lanes.mean():.3f} (max {lanes.max():.3f})"
