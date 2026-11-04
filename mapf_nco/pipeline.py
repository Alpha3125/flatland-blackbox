"""Potentials for CBS: learn edge costs on PP's delay, then use them to break CBS's ties.

    1. Scenarios: random starts and goals with a fixed PP priority order (no expert:
       nothing here needs EECBS).
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

Everything goes to ``outputs/pipeline/exp_<k>/`` (k = the smallest free number):

    pipeline_summary.md / .png       the results (tables / plots)
    pipeline_config.json             the settings of this run
    pipeline_scenarios.json.gz       the scenarios (train, validation, test)
    <map>/n<N>/models/model_<model>.pt        the learned strengths (loadable by
                                              ``main.py test`` with output_dir = this run)
    <map>/n<N>/training/training_search.*     every setting tried (json, csv, plot)
    <map>/n<N>/test/test_per_instance.csv     every method on every test scenario
    <map>/n<N>/test/test_instance_000_comparison.png   PP plain / learned / CBS plans
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
from mapf_nco.paths import checkpoint_path, test_dir, test_file, training_dir
from mapf_nco.scenarios import (Scenario, assign_priority_order, make_rng, random_scenario,
                                save_dataset)

KIND_TITLES = {"potential": "Potential", "collision": "Collision"}
VARIANTS = ("plain", "cat", "learned", "learned+cat", "weighted")
DEFAULTS = {
    "pipeline_models": ["potential", "collision"],
    "pipeline_train": 400, "pipeline_val": 100, "pipeline_test": 100,
    "pipeline_cbs_test": 50, "pipeline_time_limit": 60, "pipeline_evals": 150,
    "pipeline_box": {k: list(v) for k, v in DEFAULT_BOX.items()},
    "pipeline_plots": 3, "pipeline_variants": list(VARIANTS),
}


def setting(cfg, key):
    return cfg.get(key, DEFAULTS[key])


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
    """(key, scenario, solved, sum of costs, nodes, seconds)."""
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


# ---------------------------------------------------------------------- run
def run_pipeline(cfg, n, kinds=None, workers=None, log=print):
    from mapf_nco.sweep import next_sweep_dir, scp_command

    kinds = list(kinds or setting(cfg, "pipeline_models"))
    for k in kinds:
        strength_names(k)  # validates the name
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    grid = Grid.from_file(cfg["map_path"])
    root = next_sweep_dir(os.path.join(cfg["output_dir"], "pipeline"))
    os.makedirs(root)
    rcfg = dict(cfg, output_dir=root, pipeline_models=kinds, num_agents=n)
    for key in DEFAULTS:
        rcfg.setdefault(key, DEFAULTS[key])
    n_tr, n_val, n_te = (int(setting(cfg, k)) for k in
                         ("pipeline_train", "pipeline_val", "pipeline_test"))
    n_cbs = min(int(setting(cfg, "pipeline_cbs_test")), n_te)
    limit = float(setting(cfg, "pipeline_time_limit"))
    seed = int(cfg.get("seed", 0))
    variants = variant_list(cfg, kinds)
    with open(os.path.join(root, "pipeline_config.json"), "w") as f:
        json.dump({k: v for k, v in rcfg.items() if isinstance(v, (int, float, str, list, dict))},
                  f, indent=1)
    log(f"[pipeline] {n} agents on {grid.name}: learn {', '.join(kinds)} on PP delay "
        f"({n_tr} train / {n_val} validation), test PP on {n_te} and CBS on {n_cbs} new "
        f"scenarios ({limit:g} s limit, {len(variants)} versions). Results: {root}")
    t_start = time.time()

    # 1. scenarios
    log("[pipeline] 1/4 scenarios (random starts and goals, fixed PP priority order)")
    trval = make_scenarios(cfg, n, n_tr + n_val, 40_000_000 + 100_000 * seed, workers)
    test = make_scenarios(cfg, n, n_te, 50_000_000 + 100_000 * seed, workers)
    allsc = trval + test
    save_dataset(os.path.join(root, "pipeline_scenarios.json.gz"), grid, allsc,
                 info={"num_agents": n, "train": n_tr, "val": n_val, "test": n_te})
    tr_idx = list(range(n_tr))
    val_idx = list(range(n_tr, n_tr + n_val))
    te_idx = list(range(n_tr + n_val, len(allsc)))

    # 2. learn the strengths on PP's delay
    log(f"[pipeline] 2/4 learn the strengths on PP's delay ({setting(cfg, 'pipeline_evals')} "
        f"settings per model, {workers} worker(s))")
    ev = DelayEvaluator(cfg["map_path"], cfg["w_min"], allsc, grid, workers)
    results = {}
    try:
        for k in kinds:
            box = tuple(setting(cfg, "pipeline_box").get(k, DEFAULT_BOX[k]))
            results[k] = search(k, ev, tr_idx, val_idx, cfg.get(f"{k}_init"), box,
                                int(setting(cfg, "pipeline_evals")), seed, log)
        # 3. PP on the test scenarios
        log("[pipeline] 3/4 PP on the test scenarios")
        pp = {"plain": ev.delays("potential", np.zeros(4), te_idx)}
        for k in kinds:
            pp[k] = ev.delays(k, results[k]["theta"], te_idx)
        test_costs = {k: [ev.costs(k, results[k]["theta"], i) for i in te_idx[:n_cbs]]
                      for k in kinds}
    finally:
        ev.close()
    _save_models(rcfg, grid, n, results, log)
    _save_search(training_dir(rcfg, grid.name, n), results)

    # 4. CBS on the first n_cbs test scenarios
    log(f"[pipeline] 4/4 CBS on {n_cbs} test scenarios, {len(variants)} versions, "
        f"{limit:g} s limit each")
    tasks = []
    for j in range(n_cbs):
        sc = test[j]
        for key, mode, k, _ in variants:
            w = None if k is None else test_costs[k][j]
            tasks.append((key, mode, j, list(sc.starts), list(sc.goals), w, limit,
                          float(cfg["w_min"]), cfg["map_path"]))
    cbs = {key: [None] * n_cbs for key, *_ in variants}
    plans = {}
    with _pool(workers, initializer=_cbs_init, initargs=(cfg["map_path"],)) as ex:
        it = map(_cbs_task, tasks) if ex is None else ex.map(_cbs_task, tasks)
        for key, j, ok, soc, nodes, secs, paths in tqdm(it, total=len(tasks), desc="CBS",
                                                       unit="run", leave=False):
            cbs[key][j] = {"solved": ok, "soc": soc, "nodes": nodes, "time": secs}
            if key == "plain" and ok and j < int(setting(cfg, "pipeline_plots")):
                plans[j] = paths

    lbs = [allsc[i].lower_bound(grid) for i in te_idx]
    os.makedirs(test_dir(rcfg, grid.name, n), exist_ok=True)
    _write_per_instance(test_file(test_dir(rcfg, grid.name, n), "per_instance.csv"),
                        kinds, variants, pp, cbs, lbs, n_cbs)
    summary = _summary(kinds, variants, results, pp, cbs, lbs, n_cbs, limit)
    md = _markdown(grid, n, rcfg, kinds, variants, results, summary, n_tr, n_val, n_te, n_cbs,
                   limit, time.time() - t_start)
    with open(os.path.join(root, "pipeline_summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    _plot_summary(os.path.join(root, "pipeline_summary.png"), n, kinds, variants, summary,
                  cbs, n_cbs, limit)
    _comparisons(grid, rcfg, n, test, kinds, results, test_costs, plans,
                 min(int(setting(cfg, "pipeline_plots")), n_cbs, len(test)), log)
    log("\n" + md)
    log(f"[pipeline] done: {root}")
    cmd = scp_command(cfg, root)
    if cmd:
        log("[pipeline] to copy the results to your PC, run this on the PC:\n\n" + cmd + "\n")
    return root, summary


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
    _plot_search(os.path.join(tdir, "training_search.png"), results)


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


def _comparisons(grid, cfg, n, test, kinds, results, test_costs, plans, count, log):
    """test_instance_XXX_comparison.png: PP plain / PP learned / optimal CBS plans."""
    if count <= 0:
        return
    from mapf_nco.solver_pp import NoSolutionError, PrioritizedPlanner
    from mapf_nco.viz import plot_comparison

    planner = PrioritizedPlanner(grid, w_min=cfg["w_min"])
    out = test_dir(cfg, grid.name, n)
    for j in range(count):
        sc = test[j]
        lb = sc.lower_bound(grid)
        p = {"PP (orig)": planner.solve(sc.starts, sc.goals, sc.order)}
        for k in kinds:
            try:
                p[f"PP (learned {KIND_TITLES[k]})"] = planner.solve(sc.starts, sc.goals, sc.order,
                                                                    test_costs[k][j])
            except NoSolutionError:
                pass
        if plans.get(j):
            p["CBS (optimal)"] = plans[j]
        titles = {m: f"{m}: delay {sum(len(q) - 1 for q in v) - lb}, makespan "
                     f"{max(len(q) - 1 for q in v)}" for m, v in p.items()}
        weights = {k: test_costs[k][j] for k in kinds}
        header = (f"Potentials for CBS, test scenario {j}, {n} agents  ||  learned on PP delay: "
                  + "; ".join(f"{KIND_TITLES[k]}: " + ", ".join(
                      f"{nm} {t:+.2f}" for nm, t in zip(results[k]["names"], results[k]["theta"]))
                      for k in kinds))
        plot_comparison(grid, sc, p, titles, test_file(out, f"instance_{j:03d}_comparison.png"),
                        "PP (orig)", weights=weights,
                        weight_titles={k: f"learned {KIND_TITLES[k]} costs" for k in kinds},
                        header=header)
