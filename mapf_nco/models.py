"""Cost models: they produce one cost per directed edge (including waits).

* :class:`EdgeModel` (the paper's "Edge" model): one learnable cost per edge,
  initialised to the original unit cost 1.0 (or to a given cost vector, e.g.
  a trained potential model's costs for ``potential_edge``).
* :class:`PotentialModel` (PotP, see mapf_nco/potential.py): costs from a few
  learnable strengths on fixed map features (closeness to obstructions,
  single-lane passages, dead ends) plus a learnable one-way preference per row
  and per column: w = 1 + F @ theta.
* :class:`CollisionModel` (see mapf_nco/collision.py): four learnable strengths
  on per-scenario collision features (individual shortest paths colliding, and
  the original PP plan's waits and close calls): w = 1 + F(scenario) @ theta.

The Edge and potential models ignore the scenario (one cost vector for all);
the collision model's costs depend on it. All return a tensor of shape (B, M),
one row per scenario.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mapf_nco.potential import feature_matrix, init_theta
from mapf_nco.scenarios import scenario_arrays

KINDS = ("edge", "potential", "potential_edge", "collision")


class EdgeModel(nn.Module):
    """Direct per-edge parameters (the paper's "Edge" model)."""

    def __init__(self, num_edges, init=1.0):
        super().__init__()
        if np.ndim(init) == 0:
            w = torch.full((num_edges,), float(init))
        else:
            w = torch.from_numpy(np.array(init, dtype=np.float32))
        self.edge_weights = nn.Parameter(w)

    def costs(self):
        return self.edge_weights

    def forward(self, batch):
        B = batch["starts"].shape[0]
        return self.edge_weights.unsqueeze(0).expand(B, -1)


class PotentialModel(nn.Module):
    """w = 1 + F @ theta with fixed map features F (see mapf_nco/potential.py)."""

    def __init__(self, grid, init=None):
        super().__init__()
        # Fixed features, recomputed from the map (not stored in checkpoints).
        self.register_buffer("features", torch.from_numpy(feature_matrix(grid)), persistent=False)
        self.theta = nn.Parameter(torch.from_numpy(init_theta(grid, init)))

    def costs(self):
        return self.features @ self.theta + 1.0

    def forward(self, batch):
        B = batch["starts"].shape[0]
        return self.costs().unsqueeze(0).expand(B, -1)


class CollisionModel(nn.Module):
    """w = 1 + F(scenario) @ theta with the collision features of mapf_nco/collision.py.

    The features are computed from the batch (starts, goals and the priority order
    from the ranks) and cached per scenario; they need no expert.
    """

    def __init__(self, grid, init=None):
        super().__init__()
        from mapf_nco.collision import init_theta

        self.grid = grid
        self._cache = {}
        self.theta = nn.Parameter(torch.from_numpy(init_theta(init)))

    def features(self, starts, goals, ranks):
        from mapf_nco.collision import scenario_features

        order = [int(i) for i in np.argsort(ranks, kind="stable")]
        key = (tuple(int(s) for s in starts), tuple(int(g) for g in goals), tuple(order))
        if key not in self._cache:
            self._cache[key] = scenario_features(self.grid, list(key[0]), list(key[1]), order)
        return self._cache[key]

    def forward(self, batch):
        S, G, R = (batch[k].cpu().numpy() for k in ("starts", "goals", "ranks"))
        F = np.stack([self.features(S[b], G[b], R[b]) for b in range(len(S))])  # (B, M, K)
        B, M, K = F.shape
        Ft = torch.from_numpy(F.reshape(B * M, K)).to(self.theta.device)
        return (Ft @ self.theta + 1.0).reshape(B, M)


def build_model(kind, grid, init=None):
    """``init``: edge / potential_edge -> start costs (number or vector, default 1);
    potential / collision -> start strengths (dict, see potential.DEFAULT_INIT /
    collision.DEFAULT_INIT)."""
    if kind in ("edge", "potential_edge"):
        return EdgeModel(grid.num_edges, 1.0 if init is None else init)
    if kind == "potential":
        return PotentialModel(grid, init)
    if kind == "collision":
        return CollisionModel(grid, init)
    raise ValueError(f"Unknown model type: {kind} (choose from {', '.join(KINDS)})")


def make_batch(scenarios, device="cpu"):
    """Model input for a list of scenarios (the Edge and potential models only use the
    batch size; the collision model reads the starts, goals and priority ranks)."""
    starts, goals, ranks = scenario_arrays(scenarios)
    return {
        "starts": torch.from_numpy(starts).to(device),
        "goals": torch.from_numpy(goals).to(device),
        "ranks": torch.from_numpy(ranks).to(device),
    }
