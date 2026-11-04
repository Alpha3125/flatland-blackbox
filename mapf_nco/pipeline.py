"""Potentials for CBS (menu option 7): a sweep of experiments, each of which learns edge
costs on PP's delay and then uses them to break CBS's ties.

Every experiment (one per combination of the values given, like option 5):

    1. Scenarios: random starts and goals with a fixed PP priority order (no expert:
       nothing here needs EECBS). Experiments with the same agents and seed share them.
    2. Learn: the potential and/or collision strengths, by direct search on PP's
       average delay on the training scenarios, chosen on validation
       (mapf_nco/delay_search.py).
    3. Test PP on new scenarios: plain PP vs PP with the learned costs (delay).
    4. Test CBS on new scenarios. CBS always minimises the real sum of costs, so the
       delay is the optimum in every tie-breaking version; what can change is the
       search effort (nodes, time, scenarios solved within the time limit):
         CBS                         plain (first shortest path found)
         CBS + CAT                   conflict-avoidance table (standard CBS trick)
         CBS + <model> tie-break     the learned costs break ties
         CBS + CAT + <model> ...     conflicts first, then the learned costs
       and, for comparison, the original idea of planning on the learned costs:
         CBS on <model> costs        minimises the learned costs (delay can rise)

Sweepable (several values -> one experiment per combination): agents, models
(potential / collision / both), seed, evals, time_limit, train, cbs_test.

Everything goes to ``outputs/pipeline/exp_<k>/`` (k = the smallest free number):

    pipeline_summary.md / .csv / .png      all experiments side by side
    pipeline_config.json                   the settings and the values swept
    pipeline_scenarios_n<N>_seed<S>_*.json.gz   the scenarios (train, validation, test)
    exp_01_<settings that vary>/<map>/n<N>/
        models/model_<model>.pt            the learned strengths (``main.py test`` can load
                                           them with output_dir = the experiment folder)
        training/training_search.png / .json / _<model>.csv   every setting tried
        test/test_summary.md / .png        this experiment's tables and plots
        test/test_delays.png               PP delay per scenario, learned vs plain
        test/test_per_instance.csv         every method on every test scenario
        test/test_instance_000_comparison.png   PP plain / learned / optimal CBS plans + costs
        test/test_instance_000_cbs.png     the plan, nodes and time of every CBS version
        test/test_instance_000_animation.gif    (with animations on)
        tensorboard/tensorboard_pipeline/  search progress, summaries and plots
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from mapf_nco.delay_search import DEFAULT_BOX, DelayEvaluator, as_dict, search, strength_names
from mapf_nco.grid import Grid
from mapf_nco.paths import (checkpoint_path, tensorboard_dir, test_dir, test_file,
                            training_dir)
from mapf_nco.scenarios import (Scenario, assign_priority_order, load_dataset, make_rng,
                                random_scenario, save_dataset)

KIND_TITLES = {"potential": "Potential", "collision": "Collision"}
VARIANTS = ("plain", "cat", "learned", "learned+cat", "weighted")
DEFAULTS = {
    "pipeline_models": ["potential", "collision"],
    "pipeline_train": 400, "pipeline_val": 100, "pipeline_test": 100,
    "pipeline_cbs_test": 50, "pipeline_time_limit": 60, "pipeline_evals": 150,
    "pipeline_box": {k: list(v) for k, v in DEFAULT_BOX.items()},
    "pipeline_plots": None, "pipeline_variants": list(VARIANTS),
}
# sweepable name -> config key (agents and models are handled specially)
SWEEPABLE = {"agents": "agents", "models": "pipeline_models", "seed": "seed",
             "evals": "pipeline_evals", "time_limit": "pipeline_time_limit",
             "train": "pipeline_train", "cbs_test": "pipeline_cbs_test"}
SHORT = {"agents": "n", "models": "models", "seed": "seed", "evals": "evals",
         "time_limit": "limit", "train": "train", "cbs_test": "cbs"}
VERSION_ORDER = ["plain", "cat", "potential_tb", "potential_tb_cat", "potential_weighted",
                 "collision_tb", "collision_tb_cat", "collision_weighted"]


def setting(cfg, key):
    return cfg.get(key, DEFAULTS[key])


def models_of(value):
    """'both' / 'potential' / 'collision' / a list -> list of model names."""
    if isinstance(value, (list, tuple)):
        kinds = list(value)
    else:
        kinds = ["potential", "collision"] if value in (None, "both") else [value]
    for k in kinds:
        strength_names(k)  # validates
    return kinds


def variant_list(cfg, kinds):
    """[(key, tie mode or 'weighted', model kind or None, display name)]."""
    wanted = setting(cfg, "pipeline_variants")
    bad = set(wanted) - set(VARIANTS)
    if bad:
        raise ValueError(f"unknown pipeline_variants {sorted(bad)}; use {', '.join(VARIANTS)}")
    out = []
    if "plain" in wanted:
        out.append(("plain", "plain", None, "CBS"))
    if "cat" in wanted:
        out.append(("cat", "cat", None, "CBS + CAT"))
    for k in kinds:
        T = KIND_TITLES[k]
        if "learned" in wanted:
            out.append((f"{k}_tb", "learned", k, f"CBS + {T} tie-break"))
        if "learned+cat" in wanted:
            out.append((f"{k}_tb_cat", "learned+cat", k, f"CBS + CAT + {T} tie-break"))
        if "weighted" in wanted:
            out.append((f"{k}_weighted", "weighted", k, f"CBS on {T} costs"))
    if not out or out[0][0] != "plain":
        out.insert(0, ("plain", "plain", None, "CBS"))  # the reference is always run
    return out


VERSION_NAMES = {k: name for k, _, _, name in variant_list({}, ["potential", "collision"])}


# ------------------------------------------------------------------ scenarios
def _gen_task(args):
    map_path, n, seed = args
    grid = _grid(map_path)
    rng = make_rng(seed)
    sc = random_scenario(grid, n, rng, _S["component"])
    return sc.to_dict() if assign_priority_order(grid, sc, rng) else None


_S = {}


def _grid(map_path):
    if _S.get("map_path") != map_path:
        g = Grid.from_file(map_path)
        _S.update(map_path=map_path, grid=g, component=g.largest_component())
    return _S["grid"]


def make_scenarios(cfg, n, count, base_seed, workers):
    """``count`` random scenarios whose fixed priority order PP can solve."""
    out, seed = [], base_seed
    with _pool(workers) as ex:
        while len(out) < count:
            need = count - len(out)
            batch = [(cfg["map_path"], n, s) for s in range(seed, seed + need + need // 5 + 2)]
            seed += len(batch)
            res = (map(_gen_task, batch) if ex is None
                   else ex.map(_gen_task, batch, chunksize=max(1, len(batch) // (4 * workers))))
            out += [Scenario.from_dict(d) for d in res if d is not None][:need]
            if seed - base_seed > 20 * count + 100:
                raise RuntimeError(f"could not find {count} scenarios with {n} agents that PP "
                                   "solves under some priority order")
    return out


class _NoPool:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _pool(workers, **kw):
    return ProcessPoolExecutor(workers, **kw) if workers > 1 else _NoPool()


# ------------------------------------------------------------------------ CBS
def _cbs_init(map_path):
    _grid(map_path)


def _cbs_task(args):
    """(key, scenario, solved, sum of costs, nodes, seconds, paths)."""
    from mapf_nco.solver_cbs import TieBreakCBS, WeightedCBS
    from mapf_nco.solver_pp import NoSolutionError

    key, mode, j, starts, goals, w, limit, w_min, map_path = args
    grid = _grid(map_path)
    t0 = time.perf_counter()
    if mode == "weighted":
        cbs = WeightedCBS(grid, w_min=w_min, time_limit=limit)
        try:
            paths = cbs.solve(starts, goals, None, w)
        except NoSolutionError:
            paths = None
        secs = time.perf_counter() - t0
    else:
        cbs = TieBreakCBS(grid, mode, w_min=w_min, time_limit=limit)
        try:
            paths = cbs.solve(starts, goals, w)
        except NoSolutionError:
            paths = None
        secs = cbs.last_time
    soc = None if paths is None else sum(len(p) - 1 for p in paths)
    return key, j, paths is not None, soc, cbs.last_nodes, secs, paths


# --------------------------------------------------------------------- sweep
def run_pipeline(cfg, grid_values=None, workers=None, animate=False, log=print):
    """Run one experiment per combination of ``grid_values`` ({name: [values]}, names in
    SWEEPABLE; default: the config's single values). Returns (root folder, rows)."""
    from mapf_nco.sweep import _val, expand_grid, next_sweep_dir, scp_command

    grid_values = {k: list(v) for k, v in (grid_values or {}).items()}
    bad = set(grid_values) - set(SWEEPABLE)
    if bad:
        raise ValueError(f"cannot sweep {sorted(bad)}; choose from {', '.join(SWEEPABLE)}")
    grid_values.setdefault("agents", [int(cfg.get("num_agents", 25))])
    grid_values.setdefault("models", [cfg.get("pipeline_models", DEFAULTS["pipeline_models"])])
    for m in grid_values["models"]:
        models_of(m)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    grid = Grid.from_file(cfg["map_path"])
    experiments = expand_grid(grid_values)
    varying = [k for k, v in grid_values.items() if len(v) > 1]
    root = next_sweep_dir(os.path.join(cfg["output_dir"], "pipeline"))
    os.makedirs(root)
    with open(os.path.join(root, "pipeline_config.json"), "w") as f:
        json.dump({"swept": grid_values,
                   "config": {k: v for k, v in cfg.items()
                              if isinstance(v, (int, float, str, list, dict, type(None)))}},
                  f, indent=1)
    log(f"[pipeline] {len(experiments)} experiment(s) on {grid.name}"
        + (f", varying {', '.join(varying)}" if varying else "") + f". Results: {root}")
    rows, cache = [], {"scenarios": {}, "cbs": {}}
    for i, settings in enumerate(experiments, 1):
        name = f"exp_{i:02d}"
        folder = name + "".join(f"_{SHORT[k]}-{_val(_label(k, settings[k]))}" for k in varying)
        log(f"\n{'=' * 70}\n[pipeline] {name}/{len(experiments):02d}: "
            + ", ".join(f"{k}={_label(k, v)}" for k, v in settings.items()) + f"\n{'=' * 70}")
        ecfg = dict(cfg, output_dir=os.path.join(root, folder))
        for k, v in settings.items():
            if k not in ("agents", "models"):
                ecfg[SWEEPABLE[k]] = v
        kinds = models_of(settings["models"])
        ecfg["pipeline_models"] = kinds
        t0 = time.time()
        summary = run_experiment(ecfg, int(settings["agents"]), kinds, root, workers, animate,
                                 log, cache)
        rows.append(_sweep_row(name, folder, settings, kinds, summary, time.time() - t0))
        _write_sweep_summary(root, grid, rows, varying)  # updated after every experiment
    _plot_sweep(os.path.join(root, "pipeline_summary.png"), rows)
    with open(os.path.join(root, "pipeline_summary.md"), encoding="utf-8") as f:
        log("\n" + f.read())
    log(f"[pipeline] done: {root}")
    cmd = scp_command(cfg, root)
    if cmd:
        log("[pipeline] to copy the results to your PC, run this on the PC:\n\n" + cmd + "\n")
    return root, rows


def _label(key, value):
    if key == "models":
        kinds = models_of(value)
        return "both" if kinds == ["potential", "collision"] else "+".join(kinds)
    return value


def _scenarios(cfg, n, root, workers, cache, log):
    """(train + validation, test) scenarios for (agents, seed, counts), shared in the sweep."""
    n_tr, n_val, n_te = (int(setting(cfg, k)) for k in
                         ("pipeline_train", "pipeline_val", "pipeline_test"))
    seed = int(cfg.get("seed", 0))
    key = (n, seed, n_tr + n_val, n_te)
    if key not in cache["scenarios"]:
        path = os.path.join(root, f"pipeline_scenarios_n{n}_seed{seed}_{n_tr + n_val}-{n_te}"
                                  ".json.gz")
        grid = _grid(cfg["map_path"])
        if os.path.exists(path):
            allsc, _ = load_dataset(path, grid)
        else:
            log("[pipeline] scenarios (random starts and goals, fixed PP priority order)")
            allsc = (make_scenarios(cfg, n, n_tr + n_val, 40_000_000 + 100_000 * seed, workers)
                     + make_scenarios(cfg, n, n_te, 50_000_000 + 100_000 * seed, workers))
            save_dataset(path, grid, allsc, info={"num_agents": n, "seed": seed,
                                                  "train+val": n_tr + n_val, "test": n_te})
        cache["scenarios"][key] = (allsc[:n_tr + n_val], allsc[n_tr + n_val:], key)
    trval, test, key = cache["scenarios"][key]
    return trval, test, key


# ---------------------------------------------------------------- experiment
def run_experiment(cfg, n, kinds, root, workers, animate, log, cache):
    """One experiment (see the module docstring); writes into run_dir(cfg, map, n)."""
    from mapf_nco.tb import TB

    grid = Grid.from_file(cfg["map_path"])
    n_tr, n_val, n_te = (int(setting(cfg, k)) for k in
                         ("pipeline_train", "pipeline_val", "pipeline_test"))
    n_cbs = min(int(setting(cfg, "pipeline_cbs_test")), n_te)
    limit = float(setting(cfg, "pipeline_time_limit"))
    seed = int(cfg.get("seed", 0))
    plots = setting(cfg, "pipeline_plots")
    n_plots = n_cbs if plots is None else min(int(plots), n_cbs)
    variants = variant_list(cfg, kinds)
    log(f"[pipeline] {n} agents: learn {', '.join(kinds)} on PP delay ({n_tr} train / {n_val} "
        f"validation), test PP on {n_te} and CBS on {n_cbs} new scenarios ({limit:g} s limit, "
        f"{len(variants)} versions)")
    t_start = time.time()
    trval, test, skey = _scenarios(cfg, n, root, workers, cache, log)
    allsc = trval + test
    tr_idx = list(range(n_tr))
    val_idx = list(range(n_tr, n_tr + n_val))
    te_idx = list(range(n_tr + n_val, len(allsc)))
    tb = TB(tensorboard_dir(cfg, grid.name, n, "pipeline"), log)

    # learn the strengths on PP's delay, then PP on the test scenarios
    log(f"[pipeline] learn the strengths on PP's delay ({setting(cfg, 'pipeline_evals')} "
        f"settings per model, {workers} worker(s))")
    ev = DelayEvaluator(cfg["map_path"], cfg["w_min"], allsc, grid, workers)
    results = {}
    try:
        for k in kinds:
            box = tuple(setting(cfg, "pipeline_box").get(k, DEFAULT_BOX[k]))
            results[k] = search(k, ev, tr_idx, val_idx, cfg.get(f"{k}_init"), box,
                                int(setting(cfg, "pipeline_evals")), seed, log)
            for step, (_, d, best, _) in enumerate(results[k]["history"], 1):
                tb.scalar(f"search_{k}/train_delay", d, step)
                tb.scalar(f"search_{k}/best_so_far", best, step)
        log("[pipeline] PP on the test scenarios")
        pp = {"plain": ev.delays("potential", np.zeros(4), te_idx)}
        for k in kinds:
            pp[k] = ev.delays(k, results[k]["theta"], te_idx)
        test_costs = {k: [ev.costs(k, results[k]["theta"], i) for i in te_idx[:n_cbs]]
                      for k in kinds}
    finally:
        ev.close()
    _save_models(cfg, grid, n, results, log)
    search_png = _save_search(training_dir(cfg, grid.name, n), results)
    tb.image_file("training_search", search_png)

    # CBS on the first n_cbs test scenarios (plain / CAT results are shared in the sweep)
    log(f"[pipeline] CBS on {n_cbs} test scenarios, {len(variants)} versions, "
        f"{limit:g} s limit each")
    cbs = {key: [None] * n_cbs for key, *_ in variants}
    paths = {}
    tasks = []
    for j in range(n_cbs):
        sc = test[j]
        for key, mode, k, _ in variants:
            ck = (skey, limit, j, mode) if k is None else None
            if ck is not None and ck in cache["cbs"]:
                cbs[key][j], p = cache["cbs"][ck]
                if j < n_plots:
                    paths[(key, j)] = p
                continue
            w = None if k is None else test_costs[k][j]
            tasks.append((key, mode, j, list(sc.starts), list(sc.goals), w, limit,
                          float(cfg["w_min"]), cfg["map_path"]))
    with _pool(workers, initializer=_cbs_init, initargs=(cfg["map_path"],)) as ex:
        it = map(_cbs_task, tasks) if ex is None else ex.map(_cbs_task, tasks)
        for key, j, ok, soc, nodes, secs, p in tqdm(it, total=len(tasks), desc="CBS",
                                                   unit="run", leave=False):
            cbs[key][j] = {"solved": ok, "soc": soc, "nodes": nodes, "time": secs}
            mode = next(m for kk, m, _, _ in variants if kk == key)
            if key in ("plain", "cat"):
                cache["cbs"][(skey, limit, j, mode)] = (cbs[key][j], p)
            if j < n_plots:
                paths[(key, j)] = p

    # outputs
    lbs = [allsc[i].lower_bound(grid) for i in te_idx]
    tdir = test_dir(cfg, grid.name, n)
    os.makedirs(tdir, exist_ok=True)
    _write_per_instance(test_file(tdir, "per_instance.csv"), kinds, variants, pp, cbs, lbs, n_cbs)
    summary = _summary(kinds, variants, results, pp, cbs, lbs, n_cbs, limit)
    md = _markdown(grid, n, cfg, kinds, variants, results, summary, n_tr, n_val, n_te, n_cbs,
                   limit, time.time() - t_start)
    with open(test_file(tdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    png = _plot_summary(test_file(tdir, "summary.png"), n, kinds, variants, summary, cbs, n_cbs,
                        limit)
    delays_png = _plot_delays(test_file(tdir, "delays.png"), n, kinds, pp, cbs, lbs, n_cbs)
    log(f"[pipeline] plots of {n_plots} test scenario(s)")
    comps = _instance_plots(grid, cfg, n, test, kinds, results, test_costs, variants, cbs, paths,
                            lbs, n_plots, limit, animate, workers)
    tb.text("summary", md)
    tb.image_file("test_summary", png)
    tb.image_file("test_delays", delays_png)
    for j, path in enumerate(comps):
        tb.image_file(f"comparison/instance_{j:03d}", path)
    tb.close()
    log("\n" + md)
    log(f"[pipeline] experiment done: {tdir}")
    summary.update(results=results, n_cbs=n_cbs, limit=limit, variants=variants)
    return summary


# ------------------------------------------------------------------- outputs
def _save_models(cfg, grid, n, results, log):
    """models/model_<kind>.pt, loadable like a trained model (needs torch)."""
    try:
        from mapf_nco.models import build_model
        from mapf_nco.train import save_checkpoint
    except Exception:  # noqa: BLE001 - torch missing: the strengths are in training_search.json
        log("  (torch not installed: models not saved as .pt; see training_search.json)")
        return
    for k, r in results.items():
        init = as_dict(k, r["theta"])
        if k == "potential":
            init["lanes"] = 0.0
        model = build_model(k, grid, init=init)
        path = checkpoint_path(cfg, grid.name, n, k)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_checkpoint(path, model, k, {"init": init}, n, grid, "search", r["val_chosen"],
                        dict(cfg, train_method="delay_search", **{f"{k}_init": init}))


def _save_search(tdir, results):
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "training_search.json"), "w") as f:
        json.dump(results, f, indent=1)
    for k, r in results.items():  # training_search_<model>.csv: every setting tried
        with open(os.path.join(tdir, f"training_search_{k}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["setting"] + r["names"] + ["train_delay", "best_so_far", "pp_failures"])
            for i, (th, d, best, fails) in enumerate(r["history"], 1):
                w.writerow([i] + [f"{t:.4f}" for t in th] + [f"{d:.4f}", f"{best:.4f}", fails])
    return _plot_search(os.path.join(tdir, "training_search.png"), results)


def _plot_search(path, results):
    from mapf_nco.viz import plt

    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.4), squeeze=False)
    for ax, (k, r) in zip(axes[0], results.items()):
        h = r["history"]
        x = np.arange(1, len(h) + 1)
        ax.scatter(x, [d for _, d, _, _ in h], s=10, color="#9ecae1", label="setting tried")
        ax.plot(x, [b for _, _, b, _ in h], color="#08519c", label="best so far")
        ax.axhline(r["train_plain"], color="black", ls="--", lw=1, label="plain PP")
        ax.axhline(r["train_start"], color="#e6550d", ls=":", lw=1, label="starting strengths")
        lo = min(min(d for _, d, _, _ in h), r["train_plain"])
        hi = max(r["train_plain"], r["train_start"])
        ax.set_ylim(lo - 0.05 * (hi - lo + 1e-9), hi + 0.6 * (hi - lo + 1e-9))
        ax.set_xlabel("setting tried")
        ax.set_ylabel("PP delay on the training scenarios")
        th = ", ".join(f"{nm} {t:+.2f}" for nm, t in zip(r["names"], r["theta"]))
        ax.set_title(f"{KIND_TITLES[k]}: chosen {th}\nvalidation delay {r['val_chosen']:.2f} "
                     f"(plain PP {r['val_plain']:.2f})", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _write_per_instance(path, kinds, variants, pp, cbs, lbs, n_cbs):
    cols = ["instance", "lower_bound", "pp_plain_delay"] + [f"pp_{k}_delay" for k in kinds]
    for key, *_ in variants:
        cols += [f"{key}_solved", f"{key}_delay", f"{key}_nodes", f"{key}_seconds"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for j, lb in enumerate(lbs):
            row = [j, lb, pp["plain"][0][j]] + [pp[k][0][j] for k in kinds]
            for key, *_ in variants:
                r = cbs[key][j] if j < n_cbs else None
                row += (["", "", "", ""] if r is None else
                        [int(r["solved"]), "" if r["soc"] is None else r["soc"] - lb,
                         r["nodes"], f"{r['time']:.3f}"])
            w.writerow(row)


def _summary(kinds, variants, results, pp, cbs, lbs, n_cbs, limit):
    s = {"pp": {}, "cbs": {}, "kinds": kinds}
    base = pp["plain"][0]
    s["pp"]["plain"] = {"delay": float(base.mean()), "fails": 0}
    for k in kinds:
        d, fails = pp[k]
        s["pp"][k] = {"delay": float(d.mean()), "fails": fails,
                      "change": float(100 * (d.mean() - base.mean()) / base.mean())
                      if base.mean() else 0.0,
                      "better": int((d < base).sum()), "worse": int((d > base).sum())}
    keys = [v[0] for v in variants]
    common = [j for j in range(n_cbs) if all(cbs[k][j]["solved"] for k in keys)]
    s["common"] = common
    opt = [j for j in range(n_cbs) if cbs["plain"][j]["solved"]]
    s["pp"]["optimal_on"] = len(opt)
    if opt:
        s["pp"]["optimal"] = float(np.mean([cbs["plain"][j]["soc"] - lbs[j] for j in opt]))
        for k in ["plain"] + kinds:
            s["pp"][k]["on_opt"] = float(np.mean([pp[k][0][j] for j in opt]))
    for key, mode, k, name in variants:
        r = cbs[key]
        solved = [j for j in range(n_cbs) if r[j]["solved"]]
        both = [j for j in solved if cbs["plain"][j]["solved"]]
        e = {"name": name, "solved": len(solved),
             "time_all": float(np.mean([min(r[j]["time"], limit) if r[j]["solved"] else limit
                                        for j in range(n_cbs)])) if n_cbs else 0.0}
        if common:
            e["nodes_mean"] = float(np.mean([r[j]["nodes"] for j in common]))
            e["nodes_median"] = float(np.median([r[j]["nodes"] for j in common]))
            e["time_common"] = float(np.mean([r[j]["time"] for j in common]))
            e["delay_common"] = float(np.mean([r[j]["soc"] - lbs[j] for j in common]))
        p = cbs["plain"]
        e["vs_plain"] = _paired(r, p, n_cbs)
        if "cat" in cbs and key != "cat" and mode.endswith("cat"):
            e["vs_cat"] = _paired(r, cbs["cat"], n_cbs)
        e["both"] = len(both)
        e["delay_diff"] = [r[j]["soc"] - p[j]["soc"] for j in both]
        e["only_this"] = sum(r[j]["solved"] and not p[j]["solved"] for j in range(n_cbs))
        e["only_plain"] = sum(p[j]["solved"] and not r[j]["solved"] for j in range(n_cbs))
        s["cbs"][key] = e
    return s


def _paired(r, ref, n_cbs):
    """Nodes of version ``r`` against ``ref`` on the scenarios both solved: fewer / same /
    more, and the typical ratio (geometric mean of nodes / reference nodes)."""
    both = [j for j in range(n_cbs) if r[j]["solved"] and ref[j]["solved"]]
    a = np.array([max(1, ref[j]["nodes"]) for j in both], float)
    b = np.array([max(1, r[j]["nodes"]) for j in both], float)
    return {"n": len(both), "fewer": int((b < a).sum()), "same": int((b == a).sum()),
            "more": int((b > a).sum()),
            "ratio": float(np.exp(np.mean(np.log(b / a)))) if len(both) else None}


def _fmt_paired(d):
    if not d or not d["n"]:
        return "-"
    return f"{d['fewer']} / {d['same']} / {d['more']}, ×{d['ratio']:.2f}"


def _markdown(grid, n, cfg, kinds, variants, results, s, n_tr, n_val, n_te, n_cbs, limit, secs):
    L = [f"# Potentials for CBS: {n} agents on {grid.name}", "",
         f"{n_tr} training / {n_val} validation / {n_te} test scenarios (random, fixed PP priority "
         f"order, no expert), seed {cfg.get('seed', 0)}. Run time {secs / 60:.1f} min.", "",
         "## 1. Strengths learned on PP's delay", "",
         "Direct search on the average PP delay of the training scenarios; the final strengths "
         "are the best on validation (plain PP is a candidate, so never worse there).", "",
         "| model | learned strengths | settings tried | train delay: plain PP → start → learned "
         "| validation delay: plain PP → learned |", "|---|---|---:|---|---|"]
    for k in kinds:
        r = results[k]
        th = ", ".join(f"{nm} {t:+.3f}" for nm, t in zip(r["names"], r["theta"]))
        L.append(f"| {KIND_TITLES[k]} | {th} | {r['evaluations']} | {r['train_plain']:.2f} → "
                 f"{r['train_start']:.2f} → {r['train_chosen']:.2f} | {r['val_plain']:.2f} → "
                 f"{r['val_chosen']:.2f} ({_pct(r['val_chosen'], r['val_plain'])}) |")
    plain_chosen = [KIND_TITLES[k] for k in kinds if not any(results[k]["theta"])]
    if plain_chosen:
        L += ["", f"{' and '.join(plain_chosen)}: no setting beat plain PP on validation, so the "
              "learned costs are all 1 and its PP and tie-break rows equal plain PP / plain CBS."]
    L += ["", f"## 2. PP on the {n_te} test scenarios", "",
          "| PP with | delay | vs plain PP | scenarios better / worse | PP failures |",
          "|---|---:|---:|---:|---:|",
          f"| unit costs (plain PP) | {s['pp']['plain']['delay']:.2f} | | | |"]
    for k in kinds:
        e = s["pp"][k]
        L.append(f"| learned {KIND_TITLES[k]} costs | {e['delay']:.2f} | {e['change']:+.1f}% | "
                 f"{e['better']} / {e['worse']} | {e['fails']} |")
    if s["pp"].get("optimal_on"):
        L += ["", f"On the {s['pp']['optimal_on']} of the first {n_cbs} test scenarios that CBS "
              f"solved, the optimal delay is {s['pp']['optimal']:.2f}; PP: plain "
              f"{s['pp']['plain']['on_opt']:.2f}"
              + "".join(f", {KIND_TITLES[k]} {s['pp'][k]['on_opt']:.2f}" for k in kinds) + "."]
    L += ["", f"## 3. CBS on the first {n_cbs} test scenarios ({limit:g} s limit)", "",
          "Every version except 'CBS on ... costs' minimises the real sum of costs, so its delay "
          "must equal plain CBS's; the learned costs and the conflict-avoidance table (CAT) only "
          "break ties between equally short paths. 'CBS on ... costs' minimises the learned costs "
          "instead.", "",
          "Nodes = high-level CBS nodes. 'vs plain CBS' / 'vs CBS + CAT': on the scenarios both "
          "solved, how often this version needed fewer / the same / more nodes, and the typical "
          "ratio (geometric mean of its nodes / the other's; ×0.5 = half as many).", "",
          f"| version | solved | mean time, s (timeouts count {limit:g}) | "
          f"nodes on the {len(s['common'])} scenarios all solved: mean / median "
          "| vs plain CBS: fewer / same / more, ratio | vs CBS + CAT | "
          "solved only by this / only by plain | delay vs plain CBS |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for key, mode, k, name in variants:
        e = s["cbs"][key]
        nodes = (f"{e['nodes_mean']:.1f} / {e['nodes_median']:.0f}" if "nodes_mean" in e else "-")
        vcat = _fmt_paired(e.get("vs_cat")) if "vs_cat" in e else ""
        if key == "plain":
            vs, only, dd = "", "", "(reference)"
        else:
            vs = _fmt_paired(e["vs_plain"])
            only = f"{e['only_this']} / {e['only_plain']}"
            diff = np.array(e["delay_diff"], float)
            if not len(diff):
                dd = "-"
            elif mode == "weighted":
                dd = f"+{diff.mean():.2f} on average ({int((diff > 0).sum())}/{len(diff)} longer)"
            else:
                bad = int((diff != 0).sum())
                dd = "identical" if not bad else f"DIFFERENT on {bad} (bug!)"
        L.append(f"| {name} | {e['solved']}/{n_cbs} | {e['time_all']:.2f} | {nodes} | {vs} | "
                 f"{vcat} | {only} | {dd} |")
    return "\n".join(L) + "\n"


def _pct(x, ref):
    return "n/a" if not ref else f"{100 * (x - ref) / ref:+.1f}%"


def _plot_summary(path, n, kinds, variants, s, cbs, n_cbs, limit):
    from mapf_nco.viz import plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    ax = axes[0]
    names = ["plain PP"] + [f"PP, learned {KIND_TITLES[k]}" for k in kinds]
    vals = [s["pp"]["plain"]["delay"]] + [s["pp"][k]["delay"] for k in kinds]
    bars = ax.bar(names, vals, color=["#bdbdbd"] + ["#3182bd", "#e6550d"][:len(kinds)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("PP delay on the test scenarios (lower is better)", fontsize=10)
    ax.set_ylabel("mean delay")
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for c, (key, mode, k, name) in zip(colors, variants):
        t = sorted(r["time"] for r in cbs[key] if r["solved"])
        if t:
            ax.step([0] + t + [limit], [0] + list(range(1, len(t) + 1)) + [len(t)], where="post",
                    color=c, label=f"{name} ({len(t)})", lw=2 if key == "plain" else 1.4)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(0, limit)
    ax.set_xlabel("time per scenario, s")
    ax.set_ylabel(f"scenarios solved (of {n_cbs})")
    ax.set_title("CBS: scenarios solved within a given time (higher / further left is better)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[2]
    p = cbs["plain"]
    mx = 1
    for c, (key, mode, k, name) in zip(colors, variants):
        if key == "plain" or mode == "weighted":
            continue
        pts = [(p[j]["nodes"], cbs[key][j]["nodes"]) for j in range(n_cbs)
               if p[j]["solved"] and cbs[key][j]["solved"]]
        if pts:
            x, y = zip(*pts)
            mx = max(mx, max(x), max(y))
            ax.scatter(x, y, s=16, color=c, alpha=0.75, label=name)
    ax.plot([1, mx], [1, mx], color="black", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CBS nodes, plain")
    ax.set_ylabel("CBS nodes, with tie-breaking")
    ax.set_title("Search nodes per scenario (below the line = fewer than plain CBS)", fontsize=10)
    ax.legend(fontsize=7)
    fig.suptitle(f"Potentials for CBS, {n} agents", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_delays(path, n, kinds, pp, cbs, lbs, n_cbs):
    """test_delays.png: PP's delay on every test scenario, learned vs plain (left), and the
    distribution of the change (right); optimal delay where CBS solved."""
    from mapf_nco.viz import plt

    base = pp["plain"][0]
    colors = {"potential": "#3182bd", "collision": "#e6550d"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    mx = max([base.max()] + [pp[k][0].max() for k in kinds] + [1])
    for k in kinds:
        jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(base))
        ax.scatter(base + jit, pp[k][0] + jit, s=14, alpha=0.6, color=colors[k],
                   label=f"learned {KIND_TITLES[k]} costs")
    opt = [(base[j], cbs["plain"][j]["soc"] - lbs[j]) for j in range(n_cbs)
           if cbs["plain"][j]["solved"]]
    if opt:
        ax.scatter(*zip(*opt), s=22, marker="x", color="#6a51a3", label="optimal (CBS)")
    ax.plot([0, mx], [0, mx], color="black", lw=1, ls="--")
    ax.set_xlabel("delay, plain PP")
    ax.set_ylabel("delay")
    ax.set_title("Every test scenario (below the line = lower delay than plain PP)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    diffs = [pp[k][0] - base for k in kinds]
    lo = min([d.min() for d in diffs] + [0])
    hi = max([d.max() for d in diffs] + [0])
    bins = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1)
    for k, d in zip(kinds, diffs):
        ax.hist(d, bins=bins, alpha=0.55, color=colors[k],
                label=f"{KIND_TITLES[k]}: mean {d.mean():+.2f}, better {int((d < 0).sum())}, "
                      f"worse {int((d > 0).sum())}")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("delay change vs plain PP (negative = better)")
    ax.set_ylabel("test scenarios")
    ax.set_title("Change in PP delay per scenario", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(f"PP with the learned costs, {n} agents, {len(base)} test scenarios", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_task(args):
    """Draw test_instance_XXX_comparison.png, _cbs.png (and _animation.gif) of one scenario."""
    from mapf_nco.viz import animate_comparison, plot_comparison, plot_plan_grid

    (map_path, sc, pp_plans, pp_titles, weights, weight_titles, header, cbs_plans, cbs_titles,
     cbs_ref, comp_path, cbs_path, gif_path) = args
    grid = _grid(map_path)
    plot_comparison(grid, sc, pp_plans, pp_titles, comp_path, "PP (orig)", weights=weights,
                    weight_titles=weight_titles, header=header)
    plot_plan_grid(grid, sc, cbs_plans, cbs_titles, cbs_path, cbs_ref,
                   header=header.replace("PP plans", "CBS versions"))
    if gif_path:
        animate_comparison(grid, sc, {k: v for k, v in pp_plans.items() if v is not None},
                           pp_titles, gif_path)
    return comp_path


def _instance_plots(grid, cfg, n, test, kinds, results, test_costs, variants, cbs, paths, lbs,
                    count, limit, animate, workers):
    """For each of the first ``count`` CBS test scenarios:
    test_instance_XXX_comparison.png  PP (orig) / learned PP / optimal CBS plans + the costs
    test_instance_XXX_cbs.png         every CBS version's plan, nodes and time
    test_instance_XXX_animation.gif   PP and CBS plans moving (if ``animate``)"""
    if count <= 0:
        return []
    from mapf_nco.solver_pp import NoSolutionError, PrioritizedPlanner

    planner = PrioritizedPlanner(grid, w_min=cfg["w_min"])
    out = test_dir(cfg, grid.name, n)
    strengths = "; ".join(f"{KIND_TITLES[k]}: " + ", ".join(
        f"{nm} {t:+.2f}" for nm, t in zip(results[k]["names"], results[k]["theta"]))
        for k in kinds)

    def desc(p, lb):
        return f"delay {sum(len(q) - 1 for q in p) - lb}, makespan {max(len(q) - 1 for q in p)}"

    tasks = []
    for j in range(count):
        sc, lb = test[j], lbs[j]
        pp_plans = {"PP (orig)": planner.solve(sc.starts, sc.goals, sc.order)}
        for k in kinds:
            try:
                pp_plans[f"PP (learned {KIND_TITLES[k]})"] = planner.solve(
                    sc.starts, sc.goals, sc.order, test_costs[k][j])
            except NoSolutionError:
                pp_plans[f"PP (learned {KIND_TITLES[k]})"] = None
        if paths.get(("plain", j)):
            pp_plans["CBS (optimal)"] = paths[("plain", j)]
        pp_plans = {m: p for m, p in pp_plans.items() if p is not None}
        pp_titles = {m: f"{m}: {desc(p, lb)}" for m, p in pp_plans.items()}
        cbs_plans, cbs_titles = {}, {}
        for key, _, _, name in variants:
            r = cbs[key][j]
            cbs_plans[name] = paths.get((key, j))
            cbs_titles[name] = (f"{name}\nnodes {r['nodes']}, {r['time']:.2f} s, "
                                + (desc(cbs_plans[name], lb) if r["solved"] else
                                   f"timed out ({limit:g} s)"))
        ref = next((nm for _, _, _, nm in variants if cbs_plans.get(nm)), None)
        header = (f"Potentials for CBS, test scenario {j}, {n} agents: PP plans  ||  learned on "
                  f"PP delay: {strengths}")
        tasks.append((cfg["map_path"], sc, pp_plans, pp_titles,
                      {k: test_costs[k][j] for k in kinds},
                      {k: f"learned {KIND_TITLES[k]} costs" for k in kinds}, header,
                      cbs_plans, cbs_titles, ref,
                      test_file(out, f"instance_{j:03d}_comparison.png"),
                      test_file(out, f"instance_{j:03d}_cbs.png"),
                      test_file(out, f"instance_{j:03d}_animation.gif") if animate else None))
    with _pool(workers, initializer=_cbs_init, initargs=(cfg["map_path"],)) as ex:
        it = map(_plot_task, tasks) if ex is None else ex.map(_plot_task, tasks)
        return list(tqdm(it, total=len(tasks), desc="plots", unit="scenario", leave=False))


# ------------------------------------------------------------- sweep summary
def _sweep_row(name, folder, settings, kinds, s, secs):
    row = {"experiment": name, "folder": folder,
           **{k: _label(k, v) for k, v in settings.items()}}
    for k in kinds:
        r = s["results"][k]
        row[f"{k}_strengths"] = ", ".join(f"{nm} {t:+.3f}" for nm, t in zip(r["names"],
                                                                           r["theta"]))
        row[f"{k}_val_delay"] = round(r["val_chosen"], 3)
    row["val_plain"] = round(s["results"][kinds[0]]["val_plain"], 3)
    row["pp_plain"] = round(s["pp"]["plain"]["delay"], 3)
    for k in kinds:
        row[f"pp_{k}"] = round(s["pp"][k]["delay"], 3)
        row[f"pp_{k}_change_pct"] = round(s["pp"][k]["change"], 1)
    if s["pp"].get("optimal_on"):
        row["optimal_delay"] = round(s["pp"]["optimal"], 3)
    row["cbs_scenarios"] = s["n_cbs"]
    row["time_limit"] = s["limit"]
    for key, *_ in s["variants"]:
        e = s["cbs"][key]
        row[f"{key}_solved"] = e["solved"]
        row[f"{key}_mean_time"] = round(e["time_all"], 3)
        if key != "plain" and e["vs_plain"]["ratio"] is not None:
            row[f"{key}_vs_plain"] = round(e["vs_plain"]["ratio"], 3)
            row[f"{key}_fewer_than_plain"] = f"{e['vs_plain']['fewer']}/{e['vs_plain']['n']}"
        if e.get("vs_cat") and e["vs_cat"]["ratio"] is not None:
            row[f"{key}_vs_cat"] = round(e["vs_cat"]["ratio"], 3)
            row[f"{key}_fewer_than_cat"] = f"{e['vs_cat']['fewer']}/{e['vs_cat']['n']}"
        if key != "plain":
            diff = np.array(e["delay_diff"], float)
            row[f"{key}_delay_mismatch"] = int((diff != 0).sum()) if "weighted" not in key \
                else round(float(diff.mean()), 3) if len(diff) else ""
    row["minutes"] = round(secs / 60, 1)
    return row


def _write_sweep_summary(root, grid, rows, varying):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(root, "pipeline_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    sets = ["agents", "models"] + [k for k in varying if k not in ("agents", "models")]
    kinds = [k for k in KIND_TITLES if any(f"pp_{k}" in r for r in rows)]
    versions = [v for v in VERSION_ORDER if any(f"{v}_solved" in r for r in rows)]

    def cell(r, k, fmt="{}"):
        return fmt.format(r[k]) if k in r and r[k] != "" else "-"

    def table(head, rows_):
        return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + \
            ["| " + " | ".join(c) + " |" for c in rows_]

    def exp_cells(r):
        return [r["experiment"][4:]] + [str(r.get(k, "-")) for k in sets]

    L = [f"# Potentials for CBS: {len(rows)} experiment(s) on {grid.name}", "",
         "Each experiment learns the strengths on PP's delay (training scenarios, chosen on "
         "validation), then tests PP and CBS on new scenarios. Details and plots of one "
         "experiment: `<folder>/<map>/n<N>/test/test_summary.md` (folder in the last table).",
         "", "## Learned strengths and PP delay on the test scenarios", ""]
    L += table(["exp"] + sets + [f"{KIND_TITLES[k]} strengths" for k in kinds]
               + ["PP delay: plain"] + [f"{KIND_TITLES[k]} (change)" for k in kinds]
               + ["optimal (CBS-solved)"],
               [exp_cells(r) + [cell(r, f"{k}_strengths") for k in kinds]
                + [cell(r, "pp_plain", "{:.2f}")]
                + [f"{r[f'pp_{k}']:.2f} ({r[f'pp_{k}_change_pct']:+.1f}%)" if f"pp_{k}" in r
                   else "-" for k in kinds] + [cell(r, "optimal_delay", "{:.2f}")] for r in rows])
    L += ["", "## CBS: scenarios solved within the time limit", ""]
    L += table(["exp"] + sets + ["limit, s"] + [VERSION_NAMES[v] for v in versions],
               [exp_cells(r) + [str(r["time_limit"])]
                + [f"{r[f'{v}_solved']}/{r['cbs_scenarios']}" if f"{v}_solved" in r else "-"
                   for v in versions] for r in rows])
    L += ["", "## CBS: typical nodes vs plain CBS (geometric mean of the ratio; ×0.5 = half as "
          "many; in brackets: scenarios with fewer nodes / both solved)", ""]
    vs = [v for v in versions if v != "plain"]
    L += table(["exp"] + sets + [VERSION_NAMES[v] for v in vs],
               [exp_cells(r) + [f"×{r[f'{v}_vs_plain']:.2f} ({r[f'{v}_fewer_than_plain']})"
                                if f"{v}_vs_plain" in r else "-" for v in vs] for r in rows])
    cats = [v for v in versions if v.endswith("_tb_cat")]
    if cats:
        L += ["", "## CBS: typical nodes of the learned tie-break on top of CAT vs CBS + CAT "
              "alone", ""]
        L += table(["exp"] + sets + [VERSION_NAMES[v] for v in cats],
                   [exp_cells(r) + [f"×{r[f'{v}_vs_cat']:.2f} ({r[f'{v}_fewer_than_cat']})"
                                    if f"{v}_vs_cat" in r else "-" for v in cats] for r in rows])
    bad = [f"exp {r['experiment'][4:]}: {VERSION_NAMES[v]} on {r[f'{v}_delay_mismatch']}"
           for r in rows for v in versions
           if "weighted" not in v and v != "plain" and r.get(f"{v}_delay_mismatch")]
    L += ["", "Delay check: every tie-break version had the same delay as plain CBS on every "
          "scenario both solved." if not bad else
          "Delay check FAILED (a tie-break version changed the delay; this is a bug): "
          + "; ".join(bad)]
    L += ["", "## Folders and run time", ""]
    L += table(["exp", "folder", "minutes"], [[r["experiment"][4:], r["folder"], str(r["minutes"])]
                                              for r in rows])
    with open(os.path.join(root, "pipeline_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _plot_sweep(path, rows):
    """pipeline_summary.png: PP delay change, CBS solved and typical nodes per experiment."""
    from mapf_nco.viz import plt

    names = [r["experiment"][4:] for r in rows]
    x = np.arange(len(rows))
    kinds = [k for k in KIND_TITLES if any(f"pp_{k}" in r for r in rows)]
    versions = [v for v in VERSION_ORDER if any(f"{v}_solved" in r for r in rows)]
    colors = dict(zip(VERSION_ORDER, plt.cm.tab10(np.linspace(0, 1, 10))))
    fig, axes = plt.subplots(1, 3, figsize=(max(16, 2.2 * len(rows) + 10), 6.2))

    def bars(ax, keys, get, label, color):
        width = 0.8 / max(1, len(keys))
        for i, k in enumerate(keys):
            vals = [get(r, k) for r in rows]
            ax.bar(x + (i - (len(keys) - 1) / 2) * width,
                   [np.nan if v is None else v for v in vals], width, label=label(k),
                   color=color(k))
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel("experiment (see pipeline_summary.md)")

    bars(axes[0], kinds, lambda r, k: r.get(f"pp_{k}_change_pct"),
         lambda k: f"learned {KIND_TITLES[k]} costs",
         lambda k: {"potential": "#3182bd", "collision": "#e6550d"}[k])
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_ylabel("PP delay change vs plain PP, %")
    axes[0].set_title("PP on the test scenarios (lower is better)", fontsize=10)
    axes[0].legend(fontsize=8)
    bars(axes[1], versions,
         lambda r, k: (r[f"{k}_solved"] / r["cbs_scenarios"]) if f"{k}_solved" in r
         and r["cbs_scenarios"] else None, lambda k: VERSION_NAMES[k], lambda k: colors[k])
    axes[1].set_ylabel("fraction of CBS test scenarios solved")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("CBS: solved within the time limit (higher is better)", fontsize=10)
    vs = [v for v in versions if v != "plain"]
    bars(axes[2], vs, lambda r, k: r.get(f"{k}_vs_plain"), lambda k: VERSION_NAMES[k],
         lambda k: colors[k])
    axes[2].axhline(1, color="black", lw=1)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("typical nodes / plain CBS's nodes")
    axes[2].set_title("CBS: search size vs plain CBS (lower is better)", fontsize=10)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False,
               title="CBS versions (middle and right)", title_fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
