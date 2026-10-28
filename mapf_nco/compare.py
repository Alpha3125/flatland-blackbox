"""Shared helpers for testing: load trained models, plan every method on a
scenario, and describe the training hyperparameters for plot headers."""

from __future__ import annotations

import os
import time

import numpy as np

from mapf_nco.solver_pp import NoSolutionError

REF = "PP (orig)"
LABELS = {"edge": "PP (Edge)", "gnn": "PP (GNN)"}
WEIGHT_LABELS = {"edge": "Edge model costs", "gnn": "GNN model costs"}


def load_models(cfg, grid, n, kinds, log=print):
    """Trained models for ``n`` agents; each gets ``.ckpt_info`` (epoch, config)."""
    from mapf_nco.paths import checkpoint_path

    models = {}
    for kind in kinds:
        ckpt = checkpoint_path(cfg, grid.name, n, kind)
        if os.path.exists(ckpt):
            from mapf_nco.train import load_model  # needs torch only when a model exists

            model, info = load_model(ckpt, grid)
            model.ckpt_info = info
            models[kind] = model
            log(f"loaded {kind} model ({ckpt}, best epoch {info.get('epoch')})")
        else:
            log(f"no {kind} model at {ckpt} (run an experiment first); skipping")
    return models


def predict(model, sc, w_min):
    """Edge costs a model gives one scenario, as PP will use them."""
    import torch

    from mapf_nco.models import make_batch

    with torch.no_grad():
        w = model(make_batch([sc]))[0].double().numpy()
    return np.maximum(w, w_min)


def plan_all_timed(planner, sc, models):
    """Plans of every method for one scenario, timings, and the models' edge costs.

    Returns (plans, times, weights). A learned PP that fails maps to None. The
    original PP runs with explicit unit costs so it uses the same heuristic code
    path as the learned models (fair timing; the plan is identical to the stored
    one). EECBS and CBS plans come from the dataset.
    """
    t0 = time.perf_counter()
    plans = {REF: planner.solve(sc.starts, sc.goals, sc.order,
                                np.ones(planner.grid.num_edges))}
    times = {REF: time.perf_counter() - t0}
    weights = {}
    for kind, model in models.items():
        t0 = time.perf_counter()
        w = predict(model, sc, planner.w_min)
        try:
            plans[LABELS[kind]] = planner.solve(sc.starts, sc.goals, sc.order, w)
        except NoSolutionError:
            plans[LABELS[kind]] = None
        times[LABELS[kind]] = time.perf_counter() - t0
        weights[kind] = w
    if sc.expert_paths:
        plans["Expert"] = sc.expert_paths
    if sc.cbs_paths:
        plans["CBS"] = sc.cbs_paths
    return plans, times, weights


def hparams_header(models, trained_n):
    """One-line summary of the training hyperparameters of the loaded models."""
    if not models:
        return "No trained models: original PP vs EECBS/CBS only"
    any_info = next(iter(models.values())).ckpt_info
    c = any_info.get("config", {})
    parts = [f"trained on {trained_n} agents",
             f"epochs {c.get('epochs')}", f"batch {c.get('batch_size')}",
             f"lr {c.get('lr')}", f"lambda {c.get('lam')}",
             f"loss {c.get('loss_reduction', 'mean')}", f"w_min {c.get('w_min')}",
             f"val {int(round(100 * c.get('val_fraction', 0.2)))}%", f"seed {c.get('seed')}"]
    if "gnn" in models:
        g = models["gnn"].ckpt_info.get("model_kwargs", {})
        parts.append(f"GNN hidden {g.get('hidden')}, freqs {g.get('num_freqs')}, "
                     f"id {g.get('id_dim')}, mlp {g.get('mlp_hidden')}")
    parts.append("best epoch " + ", ".join(
        f"{k}: {m.ckpt_info.get('epoch')}" for k, m in models.items()))
    return " | ".join(parts)
