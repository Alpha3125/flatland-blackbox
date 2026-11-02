"""Which downstream planner(s) the learned edge costs are used by.

``cfg["planner"]``:
  "pp"    Prioritized Planning (the paper)
  "cbs"   Conflict-Based Search with the learned costs (mapf_nco.solver_cbs.WeightedCBS)
  "both"  train and test a separate Edge and GNN model for each (default)

Model ids: "edge" / "gnn" are the PP models (same files as before), "edge_cbs" /
"gnn_cbs" the CBS ones.
"""

from __future__ import annotations

PLANNER_CHOICES = ("pp", "cbs", "both")
KINDS = ("edge", "gnn")
MODEL_IDS = ("edge", "gnn", "edge_cbs", "gnn_cbs")
LABELS = {"edge": "PP (Edge)", "gnn": "PP (GNN)", "edge_cbs": "CBS (Edge)", "gnn_cbs": "CBS (GNN)"}
WEIGHT_LABELS = {"edge": "PP Edge model costs", "gnn": "PP GNN model costs",
                 "edge_cbs": "CBS Edge model costs", "gnn_cbs": "CBS GNN model costs"}


def planners_of(setting):
    setting = setting or "both"
    if setting not in PLANNER_CHOICES:
        raise ValueError(f"planner must be one of {PLANNER_CHOICES}, not {setting!r}")
    return ["pp", "cbs"] if setting == "both" else [setting]


def model_id(kind, planner):
    return kind if planner == "pp" else f"{kind}_{planner}"


def split_id(mid):
    """'gnn_cbs' -> ('gnn', 'cbs'); 'edge' -> ('edge', 'pp')."""
    kind, _, planner = mid.partition("_")
    return kind, planner or "pp"


def training_runs(cfg, kinds=KINDS):
    """(kind, planner) pairs to train for this config, PP first."""
    return [(k, p) for p in planners_of(cfg.get("planner", "both")) for k in kinds]


def model_ids(cfg, kinds=KINDS):
    return [model_id(k, p) for k, p in training_runs(cfg, kinds)]
