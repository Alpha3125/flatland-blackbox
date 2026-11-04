"""Cost models: they produce one cost per directed edge (including waits).

* :class:`EdgeModel` (the paper's "Edge" model): one learnable cost per edge,
  initialised to the original unit cost 1.0 (or to a given cost vector, e.g.
  a trained potential model's costs for ``potential_edge``).
* :class:`PotentialModel` (PotP, see mapf_nco/potential.py): costs from a few
  learnable strengths on fixed map features (closeness to obstructions,
  single-lane passages, dead ends) plus a learnable one-way preference per row
  and per column: w = 1 + F @ theta.

Neither looks at the scenario's starts and goals: the same cost vector is used
for every scenario. Both return a tensor of shape (B, M), one row per scenario.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mapf_nco.potential import feature_matrix, init_theta
from mapf_nco.scenarios import scenario_arrays

KINDS = ("edge", "potential", "potential_edge")


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


def build_model(kind, grid, init=None):
    """``init``: edge / potential_edge -> start costs (number or vector, default 1);
    potential -> start strengths (dict, see potential.DEFAULT_INIT)."""
    if kind in ("edge", "potential_edge"):
        return EdgeModel(grid.num_edges, 1.0 if init is None else init)
    if kind == "potential":
        return PotentialModel(grid, init)
    raise ValueError(f"Unknown model type: {kind} (choose from {', '.join(KINDS)})")


def make_batch(scenarios, device="cpu"):
    """Model input for a list of scenarios (the models only use the batch size)."""
    starts, goals, ranks = scenario_arrays(scenarios)
    return {
        "starts": torch.from_numpy(starts).to(device),
        "goals": torch.from_numpy(goals).to(device),
        "ranks": torch.from_numpy(ranks).to(device),
    }
