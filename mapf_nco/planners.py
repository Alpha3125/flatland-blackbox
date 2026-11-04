"""Which models are trained, and for which downstream planner(s).

``cfg["model"]``: how the edge costs are learned
  "edge"            the paper's Edge model: one learned cost per edge (default)
  "potential"       PotP: a few learned strengths on map features (obstructions,
                    single-lane passages, dead ends) + one-way lanes; see potential.py
  "potential_edge"  train the potential model, then start the Edge model from its
                    costs and train per edge (both models are kept)
  "collision"       per scenario: dearer into cells where collisions happen (individual
                    shortest paths, the original PP plan's waits and close calls),
                    cheaper into clean cells; four learned strengths; see collision.py

``cfg["planner"]``: who uses the costs
  "pp"    Prioritized Planning (the paper)
  "cbs"   Conflict-Based Search with the learned costs (mapf_nco.solver_cbs.WeightedCBS)
  "both"  a separate model for each (default)

Model ids (also the checkpoint file names): <kind> for PP, <kind>_cbs for CBS,
e.g. edge, potential_edge_cbs.
"""

from __future__ import annotations

PLANNER_CHOICES = ("pp", "cbs", "both")
MODEL_CHOICES = ("edge", "potential", "potential_edge", "collision")
KINDS = MODEL_CHOICES
KIND_LABELS = {"edge": "Edge", "potential": "Potential", "potential_edge": "Potential+Edge",
               "collision": "Collision"}


def model_id(planner, kind="edge"):
    return kind if planner == "pp" else f"{kind}_{planner}"


def planner_of(mid):
    """'edge_cbs' -> 'cbs'; 'potential' -> 'pp'."""
    return "cbs" if mid.endswith("_cbs") else "pp"


def kind_of(mid):
    return mid[:-4] if mid.endswith("_cbs") else mid


MODEL_IDS = tuple(model_id(p, k) for p in ("pp", "cbs") for k in KINDS)
LABELS = {m: f"{planner_of(m).upper()} ({KIND_LABELS[kind_of(m)]})" for m in MODEL_IDS}
WEIGHT_LABELS = {m: f"{planner_of(m).upper()} {KIND_LABELS[kind_of(m)]} model costs"
                 for m in MODEL_IDS}


def planners_of(setting):
    setting = setting or "both"
    if setting not in PLANNER_CHOICES:
        raise ValueError(f"planner must be one of {PLANNER_CHOICES}, not {setting!r}")
    return ["pp", "cbs"] if setting == "both" else [setting]


def kinds_of(setting):
    """Models to train, in order: potential_edge needs the potential model first."""
    setting = setting or "edge"
    if setting not in MODEL_CHOICES:
        raise ValueError(f"model must be one of {MODEL_CHOICES}, not {setting!r}")
    return ["potential", "potential_edge"] if setting == "potential_edge" else [setting]


def training_runs(cfg):
    """(kind, planner) pairs to train for this config, PP first."""
    return [(k, p) for p in planners_of(cfg.get("planner", "both"))
            for k in kinds_of(cfg.get("model", "edge"))]


def model_ids(cfg):
    """Model ids trained by this config."""
    return [model_id(p, k) for k, p in training_runs(cfg)]
