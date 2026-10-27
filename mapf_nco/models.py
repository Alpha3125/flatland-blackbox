"""Neural cost-shaping models N_theta(x) -> w (paper Sec. 3.3 and 4).

Two variants are compared in the paper:

* :class:`EdgeModel` — one learnable scalar per directed edge, initialised to
  the original unit cost 1.0. It ignores start/goal inputs.
* :class:`GNNModel` — a two-layer GraphSAGE network over the grid. Node
  features encode a demand signal (starts vs. goals), sinusoidal positional
  embeddings and a small agent-ID embedding; each edge weight is produced by a
  lightweight MLP over its two endpoint embeddings.

Both return a tensor of shape (B, M) with one cost per directed edge
(including wait self-loops) for each scenario in the batch. The GNN output is
parametrised so that it starts at exactly w = 1 everywhere, i.e. training starts
from the original PP.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mapf_nco.scenarios import scenario_arrays

_SOFTPLUS_ONE = math.log(math.e - 1.0)  # softplus(_SOFTPLUS_ONE) == 1


class EdgeModel(nn.Module):
    """Direct per-edge parameters (the paper's "Edge" model)."""

    def __init__(self, num_edges, init=1.0):
        super().__init__()
        self.edge_weights = nn.Parameter(torch.full((num_edges,), float(init)))

    def forward(self, batch):
        B = batch["starts"].shape[0]
        return self.edge_weights.unsqueeze(0).expand(B, -1)


def sinusoidal_positions(coords, height, width, num_freqs):
    """Sin/cos embeddings of (row, col), shape (N, 4 * num_freqs)."""
    rc = coords.astype(np.float64) / np.array([max(height, 1), max(width, 1)])
    freqs = (2.0 ** np.arange(num_freqs)) * math.pi
    ang = rc[:, :, None] * freqs[None, None, :]  # (N, 2, F)
    emb = np.concatenate([np.sin(ang), np.cos(ang)], axis=2).reshape(len(coords), -1)
    return emb.astype(np.float32)


class SAGELayer(nn.Module):
    """GraphSAGE layer with mean aggregation (Hamilton et al., 2017)."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_neigh = nn.Linear(d_in, d_out, bias=False)

    def forward(self, h, src, dst, inv_deg):
        # h: (B, N, d). Mean over neighbours u of v for every edge u->v.
        agg = torch.zeros_like(h)
        agg.index_add_(1, dst, h[:, src, :])
        agg = agg * inv_deg.view(1, -1, 1)
        return self.lin_self(h) + self.lin_neigh(agg)


class GNNModel(nn.Module):
    """Two-layer GraphSAGE edge-cost predictor (the paper's "GNN" model)."""

    def __init__(self, grid, hidden=64, num_freqs=6, id_dim=8, max_agents=128,
                 mlp_hidden=64):
        super().__init__()
        self.max_agents = max_agents
        pos = sinusoidal_positions(grid.coords, grid.height, grid.width, num_freqs)
        deg = (grid.degree.astype(np.float32) / 4.0)[:, None]  # static obstacle context
        self.register_buffer("static_feats", torch.from_numpy(np.concatenate([pos, deg], 1)))
        moves = grid.move_edges
        self.register_buffer("msg_src", torch.from_numpy(grid.edge_src[moves]))
        self.register_buffer("msg_dst", torch.from_numpy(grid.edge_dst[moves]))
        inv_deg = 1.0 / np.maximum(grid.degree, 1).astype(np.float32)
        self.register_buffer("inv_deg", torch.from_numpy(inv_deg))
        self.register_buffer("edge_src", torch.from_numpy(grid.edge_src))
        self.register_buffer("edge_dst", torch.from_numpy(grid.edge_dst))
        self.num_nodes = grid.num_nodes

        # Agent ID = the agent's rank in the PP priority order, which is what
        # distinguishes agents from PP's point of view.
        self.id_emb = nn.Embedding(max_agents, id_dim)
        d_in = self.static_feats.shape[1] + 2 + 2 * id_dim
        self.sage1 = SAGELayer(d_in, hidden)
        self.sage2 = SAGELayer(hidden, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1)
        )
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)

    def node_features(self, batch):
        starts, goals, ranks = batch["starts"], batch["goals"], batch["ranks"]
        B, n = starts.shape
        N = self.num_nodes
        dev = self.static_feats.device
        ones = torch.ones(B, n, device=dev)
        start_count = torch.zeros(B, N, device=dev).scatter_add(1, starts, ones)
        goal_count = torch.zeros(B, N, device=dev).scatter_add(1, goals, ones)
        demand = torch.stack([start_count, goal_count], dim=-1)  # (B, N, 2)
        ids = self.id_emb(ranks.clamp(max=self.max_agents - 1))  # (B, n, id_dim)
        id_start = torch.zeros(B, N, ids.shape[-1], device=dev)
        id_goal = torch.zeros_like(id_start)
        idx = lambda t: t.unsqueeze(-1).expand(-1, -1, ids.shape[-1]).contiguous()  # noqa: E731
        id_start = id_start.scatter_add(1, idx(starts), ids)
        id_goal = id_goal.scatter_add(1, idx(goals), ids)
        static = self.static_feats.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([static, demand, id_start, id_goal], dim=-1)

    def forward(self, batch):
        h = self.node_features(batch)
        h = F.relu(self.sage1(h, self.msg_src, self.msg_dst, self.inv_deg))
        h = F.relu(self.sage2(h, self.msg_src, self.msg_dst, self.inv_deg))
        pair = torch.cat([h[:, self.edge_src, :], h[:, self.edge_dst, :]], dim=-1)
        raw = self.edge_mlp(pair).squeeze(-1)  # (B, M)
        return F.softplus(raw + _SOFTPLUS_ONE)  # == 1 at initialisation, always > 0


def build_model(kind, grid, **kwargs):
    if kind == "edge":
        return EdgeModel(grid.num_edges)
    if kind == "gnn":
        return GNNModel(grid, **kwargs)
    raise ValueError(f"Unknown model type: {kind}")


def make_batch(scenarios, device="cpu"):
    starts, goals, ranks = scenario_arrays(scenarios)
    return {
        "starts": torch.from_numpy(starts).to(device),
        "goals": torch.from_numpy(goals).to(device),
        "ranks": torch.from_numpy(ranks).to(device),
    }
