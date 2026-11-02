"""Shared helpers for testing: load trained models, plan every method on a
scenario, and describe the training hyperparameters for plot headers."""

from __future__ import annotations

import os
import time

import numpy as np

from mapf_nco.solver_pp import NoSolutionError

from mapf_nco.planners import LABELS, MODEL_IDS, WEIGHT_LABELS, split_id  # noqa: F401 (re-exported)

REF = "PP (orig)"


def load_models(cfg, grid, n, kinds=MODEL_IDS, log=print):
    """Trained models for ``n`` agents; each gets ``.ckpt_info`` (epoch, config).

    ``kinds``: model ids (edge, gnn = PP; edge_cbs, gnn_cbs = CBS) or plain
    edge/gnn, which select both planners' models. Missing ones are skipped.
    """
    from mapf_nco.paths import checkpoint_path

    ids = [m for m in MODEL_IDS if m in kinds or split_id(m)[0] in kinds]
    models, missing = {}, []
    for mid in ids:
        ckpt = checkpoint_path(cfg, grid.name, n, mid)
        if os.path.exists(ckpt):
            from mapf_nco.train import load_model  # needs torch only when a model exists

            model, info = load_model(ckpt, grid)
            model.ckpt_info = info
            models[mid] = model
            log(f"loaded {LABELS[mid]} model ({ckpt}, best epoch {info.get('epoch')})")
        else:
            missing.append(LABELS[mid])
    if missing:
        log("not trained (skipped): " + ", ".join(missing))
    return models


def predict(model, sc, w_min):
    """Edge costs a model gives one scenario, as PP will use them."""
    import torch

    from mapf_nco.models import make_batch

    with torch.no_grad():
        w = model(make_batch([sc]))[0].double().numpy()
    return np.maximum(w, w_min)


def model_method(model):
    return getattr(model, "ckpt_info", {}).get("config", {}).get("method", "static")


def plan_all_timed(planner, sc, models, cbs=None):
    """Plans of every method for one scenario, timings, and the models' edge costs.

    Returns (plans, times, weights, seq_weights). A learned PP that fails maps
    to None. Sequential models (method seq_*) are called before every agent:
    ``weights`` then holds the mean over agents and ``seq_weights[kind]`` the
    cost vector each agent planned with (indexed by agent). The
    original PP runs with explicit unit costs so it uses the same heuristic code
    path as the learned models (fair timing; the plan is identical to the stored
    one). EECBS plans come from the dataset. CBS models plan with ``cbs`` (a
    WeightedCBS); then the original CBS is run here too (unit costs, timed, same
    time limit) so the times compare fairly; otherwise, and if it times out, the
    CBS plan comes from the dataset.
    """
    t0 = time.perf_counter()
    plans = {REF: planner.solve(sc.starts, sc.goals, sc.order,
                                np.ones(planner.grid.num_edges))}
    times = {REF: time.perf_counter() - t0}
    weights, seq_weights = {}, {}
    cbs_models = {k: m for k, m in models.items() if split_id(k)[1] == "cbs"}
    if cbs is not None and cbs_models:
        t0 = time.perf_counter()
        try:
            plans["CBS"] = cbs.solve(sc.starts, sc.goals)
            times["CBS"] = time.perf_counter() - t0
        except NoSolutionError:
            pass
    for kind, model in models.items():
        t0 = time.perf_counter()
        if kind in cbs_models:
            w = predict(model, sc, planner.w_min)
            try:
                plans[LABELS[kind]] = cbs.solve(sc.starts, sc.goals, None, w)
            except (NoSolutionError, AttributeError):
                plans[LABELS[kind]] = None
            times[LABELS[kind]] = time.perf_counter() - t0
            weights[kind] = w
            continue
        if model_method(model) != "static":
            from mapf_nco.sequential import cost_fn_for, sequential_pp

            fn = cost_fn_for(model, sc)
            try:
                paths, W, _ = sequential_pp(planner, sc.starts, sc.goals, sc.order, fn)
                plans[LABELS[kind]] = paths
            except NoSolutionError:
                plans[LABELS[kind]] = None
                W = []
            times[LABELS[kind]] = time.perf_counter() - t0
            if W and all(w is not None for w in W):
                W = [np.maximum(w, planner.w_min) for w in W]
                seq_weights[kind] = W
                weights[kind] = np.mean(W, axis=0)
            else:  # failed part-way: show the costs of the first agent
                weights[kind] = predict(model, sc, planner.w_min)
            continue
        w = predict(model, sc, planner.w_min)
        try:
            plans[LABELS[kind]] = planner.solve(sc.starts, sc.goals, sc.order, w)
        except NoSolutionError:
            plans[LABELS[kind]] = None
        times[LABELS[kind]] = time.perf_counter() - t0
        weights[kind] = w
    if sc.expert_paths:
        plans["Expert"] = sc.expert_paths
    if sc.cbs_paths and "CBS" not in plans:
        plans["CBS"] = sc.cbs_paths
    order = [REF] + [LABELS[k] for k in models] + ["Expert", "CBS"]
    plans = {k: plans[k] for k in order if k in plans}
    return plans, times, weights, seq_weights


def hparams_header(models, trained_n):
    """One-line summary of the training hyperparameters of the loaded models."""
    if not models:
        return "No trained models: original PP vs EECBS/CBS only"
    any_info = next(iter(models.values())).ckpt_info
    c = any_info.get("config", {})
    obj = c.get("objective", "usage")
    if obj == "makespan":
        obj += f" (tau {c.get('makespan_tau', 1.0)})"
    elif obj == "usage+makespan":
        obj += f" (tau {c.get('makespan_tau', 1.0)}, weight {c.get('makespan_weight', 0.001)})"
    meths = sorted({model_method(m) for k, m in models.items() if split_id(k)[1] == "pp"})
    pls = [p for p in ("pp", "cbs") if any(split_id(k)[1] == p for k in models)]
    parts = [f"planner {' + '.join(p.upper() for p in pls)}"]
    if meths:
        parts.append(f"PP method {'/'.join(meths)}")
    parts += [ f"objective {obj}", f"trained on {trained_n} agents",
             f"epochs {c.get('epochs')}", f"batch {c.get('batch_size')}",
             f"lr {c.get('lr')}", f"lambda {c.get('lam')}",
             f"loss {c.get('loss_reduction', 'mean')}", f"w_min {c.get('w_min')}",
             f"val {int(round(100 * c.get('val_fraction', 0.2)))}%", f"seed {c.get('seed')}"]
    gnn = next((m for k, m in models.items() if split_id(k)[0] == "gnn"), None)
    if gnn is not None:
        g = gnn.ckpt_info.get("model_kwargs", {})
        parts.append(f"GNN hidden {g.get('hidden')}, freqs {g.get('num_freqs')}, "
                     f"id {g.get('id_dim')}, mlp {g.get('mlp_hidden')}")
    parts.append("best epoch " + ", ".join(
        f"{LABELS[k]}: {m.ckpt_info.get('epoch')}" for k, m in models.items()))
    return " | ".join(parts)
