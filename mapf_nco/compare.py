"""Shared helpers for testing: load trained models, plan every method on a
scenario, and describe the training hyperparameters for plot headers."""

from __future__ import annotations

import time

import numpy as np

from mapf_nco.planners import (KIND_LABELS, LABELS, MODEL_IDS, WEIGHT_LABELS,  # noqa: F401
                                kind_of, planner_of)
from mapf_nco.solver_pp import NoSolutionError

REF = "PP (orig)"
CBS_TIMEOUT = "CBS timed out"  # key in the times dict: the live original CBS hit its limit


def load_models(cfg, grid, n, kinds=MODEL_IDS, log=print):
    """Trained models for ``n`` agents; each gets ``.ckpt_info`` (epoch, config).

    ``kinds``: model ids (e.g. edge, potential, potential_edge_cbs). Missing ones are skipped.
    """
    from mapf_nco.paths import find_checkpoint

    models, missing = {}, []
    for mid in (m for m in MODEL_IDS if m in kinds):
        ckpt = find_checkpoint(cfg, grid.name, n, mid)
        if ckpt is not None:
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
    """Edge costs a model gives one scenario, as the planner will use them."""
    import torch

    from mapf_nco.models import make_batch

    with torch.no_grad():
        w = model(make_batch([sc]))[0].double().numpy()
    return np.maximum(w, w_min)


def plan_all_timed(planner, sc, models, cbs=None):
    """Plans of every method for one scenario, timings, and the models' edge costs.

    Returns (plans, times, weights). A learned planner that fails maps to None.
    The original PP runs with explicit unit costs so it uses the same heuristic
    code path as the learned model (fair timing; the plan is identical to the
    stored one). EECBS plans come from the dataset. CBS models plan with ``cbs``
    (a WeightedCBS); then the original CBS is run here too (unit costs, timed,
    same time limit) so the times compare fairly: its time is counted even when
    it times out (``times[CBS_TIMEOUT]`` is then set), exactly like a CBS
    model's. Otherwise, and on a timeout, the CBS plan comes from the dataset.
    """
    t0 = time.perf_counter()
    plans = {REF: planner.solve(sc.starts, sc.goals, sc.order,
                                np.ones(planner.grid.num_edges))}
    times = {REF: time.perf_counter() - t0}
    weights = {}
    cbs_models = {k for k in models if planner_of(k) == "cbs"}
    if cbs is not None and cbs_models:
        t0 = time.perf_counter()
        try:
            plans["CBS"] = cbs.solve(sc.starts, sc.goals)
        except NoSolutionError:  # same rule as the CBS models: counted as a failure
            times[CBS_TIMEOUT] = True
        times["CBS"] = time.perf_counter() - t0
    for kind, model in models.items():
        t0 = time.perf_counter()
        w = predict(model, sc, planner.w_min)
        try:
            if kind in cbs_models:
                plans[LABELS[kind]] = cbs.solve(sc.starts, sc.goals, None, w)
            else:
                plans[LABELS[kind]] = planner.solve(sc.starts, sc.goals, sc.order, w)
        except (NoSolutionError, AttributeError):  # AttributeError: no CBS planner given
            plans[LABELS[kind]] = None
        times[LABELS[kind]] = time.perf_counter() - t0
        weights[kind] = w
    if sc.expert_paths:
        plans["Expert"] = sc.expert_paths
    if sc.cbs_paths and "CBS" not in plans:
        plans["CBS"] = sc.cbs_paths
    order = [REF] + [LABELS[k] for k in models] + ["Expert", "CBS"]
    plans = {k: plans[k] for k in order if k in plans}
    return plans, times, weights


def hparams_header(models, trained_n):
    """One-line summary of the training hyperparameters of the loaded models.

    A setting that differs between the models is listed per model.
    """
    if not models:
        return "No trained models: original PP vs EECBS/CBS only"

    def objective(c):
        obj = c.get("objective", "usage")
        if obj == "makespan":
            obj += f" (tau {c.get('makespan_tau', 1.0)})"
        elif obj == "usage+makespan":
            obj += f" (tau {c.get('makespan_tau', 1.0)}, weight {c.get('makespan_weight', 0.001)})"
        return obj

    configs = {k: m.ckpt_info.get("config", {}) for k, m in models.items()}

    def setting(name, get):
        vals = {k: get(c) for k, c in configs.items()}
        if len(set(map(str, vals.values()))) == 1:
            return f"{name} {next(iter(vals.values()))}"
        return f"{name} " + ", ".join(f"{LABELS[k]}: {v}" for k, v in vals.items())

    pls = [p for p in ("pp", "cbs") if any(planner_of(k) == p for k in models)]
    kinds = [KIND_LABELS[k] for k in KIND_LABELS if any(kind_of(m) == k for m in models)]
    parts = [f"model {' / '.join(kinds)}, planner {' + '.join(p.upper() for p in pls)}",
             setting("objective", objective), f"trained on {trained_n} agents",
             setting("epochs", lambda c: c.get("epochs")),
             setting("batch", lambda c: c.get("batch_size")),
             setting("lr", lambda c: c.get("lr")), setting("lambda", lambda c: c.get("lam")),
             setting("loss", lambda c: c.get("loss_reduction", "mean")),
             setting("w_min", lambda c: c.get("w_min")),
             setting("val", lambda c: f"{int(round(100 * c.get('val_fraction', 0.2)))}%"),
             setting("seed", lambda c: c.get("seed"))]
    parts.append("best epoch " + ", ".join(
        f"{LABELS[k]}: {m.ckpt_info.get('epoch')}" for k, m in models.items()))
    for k, m in models.items():
        if kind_of(k) in ("potential", "collision"):
            if kind_of(k) == "potential":
                from mapf_nco.potential import STRENGTHS as names
            else:
                from mapf_nco.collision import FEATURES as names
            th = m.theta.detach().cpu().numpy()
            parts.append(f"{LABELS[k]}: " + ", ".join(f"{n} {th[i]:+.2f}"
                                                       for i, n in enumerate(names)))
    return " | ".join(parts)
