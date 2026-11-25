"""Sweep: a queue of experiments on one or more maps, with the same tests for each.

Maps: every experiment runs on each map of ``maps`` (default: the configured map;
the menu offers random-32-32-20, random-64-64-10 and random-64-64-20). The agent
count on each map follows ``agent_scaling`` (several values allowed), relative to
the configured map (random-32-32-20):

    count           the same number of agents on every map
    count_scale_2   times the side-length ratio  (x2 on a 64x64 map)
    count_scale_4   times the area ratio         (x4 on a 64x64 map: the same density)

Each (map, agent count) gets its own data (generated if missing) and its own
<map>/n<N>/ folder inside every experiment folder.

Start/goal positions (``positions``):

    randomized      new random starts and goals for every instance (default)
    fixed           one random start/goal configuration per (map, agent count), used by
                    every instance -- train, validation and test -- and by every
                    experiment of the sweep. The PP priority order follows from the
                    positions, so every instance is the same problem: the expert, PP and
                    CBS solve it once, and each test uses one instance. This shows
                    whether a model can learn one known problem (no generalization
                    needed). Its data is kept apart, in <data_dir>/fixed_positions/.

For every combination of the hyperparameter values given (a grid search), the
chosen model (``model``: edge / potential / potential_edge / collision / cell) is
trained from scratch, for PP, CBS or both (``planner``), and then tested on

  1. the saved test scenarios (same agent count as training), and
  2. optionally, a second set of new random scenarios with a different agent
     count (off by default: every experiment then uses the selected agent count
     only), scaled per map like the training count. This set is generated once
     per map and shared by all experiments, so they are compared fairly.

A CBS model is skipped when the original CBS can't solve most validation
scenarios in time (see mapf_nco.train.train); the tests then use the models that
were trained, and sweep_summary.md lists the skipped ones.

Everything goes to ``outputs/sweeps/exp_<k>/`` (k = the smallest free number, so deleting
exp_2 makes the next sweep exp_2 again), and each experiment to
``exp_01_<the settings that vary>/`` inside it:

    sweep_summary.md / .csv          one row per experiment and map: settings, map,
                                     agents, best epochs, validation delay, test delays
    sweep_summary.png                test delay of every experiment, one panel per map
    second_test_<map>_n<M>.json.gz   the shared second test set of each map
    exp_01/ ... exp_K/<map>/n<N>/    models/, training/, tensorboard/,
                                     test/ and test_n<M>/ (comparison plots)

When it finishes, it prints an scp command that copies the sweep folder to your PC
(``scp_dest`` in the config, default C:\\Users\\Alpha\\Downloads\\; the user and
address are detected, or set ``scp_user`` / ``scp_host``; ``scp_dest`` null turns it off).
"""

from __future__ import annotations

import csv
import itertools
import os
import re
import time

from mapf_nco.grid import Grid
from mapf_nco.planners import KIND_LABELS, MODEL_IDS, kind_of, model_ids, planner_of
from mapf_nco.scenarios import load_dataset, positions_of, save_dataset

# Hyperparameters that can be swept.
SWEEPABLE = ("model", "planner", "objective", "makespan_tau", "makespan_weight", "epochs", "batch_size",
             "lr", "lam", "loss_reduction", "w_min", "val_fraction", "seed")


_SHORT = {"loss_reduction": "loss", "makespan_tau": "tau", "makespan_weight": "mw",
          "batch_size": "batch", "val_fraction": "val"}


def _short(key):
    return _SHORT.get(key, key)


def _val(v):
    return str(v).replace("+", "-").replace("/", "-")


def next_sweep_dir(base):
    """``base/exp_<k>`` with k the smallest natural number not used yet (exp_1, exp_2, ...)."""
    used = set()
    if os.path.isdir(base):
        for name in os.listdir(base):
            m = re.fullmatch(r"exp_(\d+)", name)
            if m:
                used.add(int(m.group(1)))
    k = 1
    while k in used:
        k += 1
    return os.path.join(base, f"exp_{k}")


SCALINGS = ("count", "count_scale_2", "count_scale_4")
ALL_MAPS = ("random-32-32-20", "random-64-64-10", "random-64-64-20")


def resolve_map(cfg, name):
    """A map name (random-64-64-10) or path -> its path (looked up next to the configured map)."""
    if os.path.isfile(name):
        return name
    path = os.path.join(os.path.dirname(cfg["map_path"]) or ".",
                        name if name.endswith(".map") else name + ".map")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"map {name!r} not found (looked for {path})")
    return path


def scaled_agents(n, base, grid, scaling):
    """Agents on ``grid`` for ``n`` agents on the ``base`` map (see the module docstring)."""
    area = (grid.width * grid.height) / (base.width * base.height)
    if scaling == "count":
        return n
    if scaling == "count_scale_2":
        return max(1, round(n * area ** 0.5))
    if scaling == "count_scale_4":
        return max(1, round(n * area))
    raise ValueError(f"agent_scaling must be one of {', '.join(SCALINGS)}, not {scaling!r}")


def sweep_targets(cfg, n, maps=None, scalings=None):
    """The (map, agent count) pairs a sweep runs on, one per distinct pair:
    [{map_path, grid, agents, scalings}] in the order of ``maps``."""
    base = Grid.from_file(cfg["map_path"])
    out = {}
    for m in maps or [cfg["map_path"]]:
        path = resolve_map(cfg, m)
        g = Grid.from_file(path)
        for sc in scalings or ["count"]:
            key = (g.name, scaled_agents(n, base, g, sc))
            if key not in out:
                out[key] = {"map_path": path, "grid": g, "agents": key[1], "scalings": []}
            out[key]["scalings"].append(sc)
    return list(out.values())


def second_agent_count(n, cfg):
    """Half the agents for n >= 50, otherwise the largest configured agent count."""
    if n >= 50:
        return max(1, n // 2)
    return max([int(x) for x in cfg.get("agent_counts", [100])] + [n])


def expand_grid(grid_values):
    """{'lr': [1e-4, 5e-4], 'lam': [1.0]} -> list of dicts (cartesian product)."""
    keys = list(grid_values)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid_values[k] for k in keys))]


def apply_settings(cfg, settings):
    return {**cfg, **settings}


def run_sweep(cfg, n, grid_values, test_count=None, second_n=None, second_count=None,
              second_seed=777, animate=False, workers=None, log=print, maps=None, scalings=None):
    """Run the sweep for ``n`` agents (on the configured map; scaled on the others)."""
    from mapf_nco.evaluate import run_test, test_saved
    from mapf_nco.experts import expert_options, generate_datasets, make_scenarios
    from mapf_nco.train import run_training
    from mapf_nco.viz import plot_sweep_summary

    targets = sweep_targets(cfg, n, maps, scalings)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    positions = positions_of(cfg)
    fixed = positions == "fixed"
    if fixed:
        test_count = 1  # every test instance is the same scenario
    experiments = expand_grid(grid_values)
    root = next_sweep_dir(os.path.join(cfg["output_dir"], "sweeps"))
    varying = [k for k, v in grid_values.items() if len(v) > 1]
    os.makedirs(root)
    log(f"[sweep] {len(experiments)} experiment(s) on "
        + "; ".join(f"{t['grid'].name} with {t['agents']} agents" for t in targets)
        + (f"; second test on {second_n} agents (scaled per map)" if second_n else "")
        + f"; start/goal positions: {positions}"
        + (" (one scenario per map and agent count, tested on 1 instance)" if fixed else "")
        + f". Results: {root}")

    ename = "EECBS" if cfg.get("expert", "eecbs") == "eecbs" else "CBS"
    for t in targets:  # data of every map (generated if missing), and its second test set
        tcfg = dict(cfg, map_path=t["map_path"])
        log(f"\n[sweep] data: {t['grid'].name}, {t['agents']} agents")
        generate_datasets(dict(tcfg, augment=True), [t["agents"]], workers=workers, log=log)
        t["second_n"] = max(1, round(second_n * t["agents"] / n)) if second_n else None
        t["second"] = None
        if second_n:
            expert, opts = expert_options(tcfg, t["grid"], log)
            test_size = 1 if fixed else second_count or test_count or cfg.get("num_test", 25)
            path = os.path.join(root, f"second_test_{t['grid'].name}_n{t['second_n']}.json.gz")
            sc, _, _ = make_scenarios(tcfg, t["second_n"], test_size, 20_000_000 + second_seed,
                                      expert, opts, workers,
                                      desc=f"second test set ({t['grid'].name}, "
                                           f"{t['second_n']} agents)", fixed=fixed)
            save_dataset(path, t["grid"], sc, info={"num_agents": t["second_n"], "expert": expert,
                                                    "positions": positions})
            t["second"], _ = load_dataset(path, t["grid"])
    tags = ["test"] + (["second"] if second_n else [])

    rows = []
    for i, settings in enumerate(experiments, 1):
        name = f"exp_{i:02d}"
        folder = name + "".join(f"_{_short(k)}-{_val(settings[k])}" for k in varying)
        for t in targets:
            grid, tn = t["grid"], t["agents"]
            log(f"\n{'=' * 70}\n[sweep] {name}/{len(experiments):02d} on {grid.name}, {tn} agents: "
                + ", ".join(f"{k}={v}" for k, v in settings.items()) + f"\n{'=' * 70}")
            ecfg = apply_settings(dict(cfg, map_path=t["map_path"]), settings)
            ecfg["output_dir"] = os.path.join(root, folder)
            row = {"experiment": name, "map": grid.name, "agents": tn,
                   "scaling": "/".join(t["scalings"]), "positions": positions, **settings}
            if t["second_n"]:
                row["second_agents"] = t["second_n"]
            t0 = time.time()
            ids = run_training(ecfg, tn, workers=workers, log=log)  # skipped CBS models left out
            skipped = [m for m in model_ids(ecfg) if m not in ids]
            if skipped:
                row["not_trained"] = " / ".join(
                    f"{planner_of(m).upper()} {KIND_LABELS[kind_of(m)]}" for m in skipped)
            s1 = test_saved(ecfg, tn, test_count, ids, plots=True, animate=animate, log=log)
            summaries = [s1]
            if t["second"] is not None:
                summaries.append(run_test(ecfg, grid, tn, t["second"], ename,
                                          (f"the fixed scenario of {t['second_n']} agents"
                                           if fixed else f"{len(t['second'])} new random "
                                           f"scenarios, {t['second_n']} agents"),
                                          ids, True, animate, log,
                                          out_name=f"test_n{t['second_n']}"))
            row.update(_best_epochs(ecfg, grid, tn, ids))
            for tag, s in zip(tags, summaries):
                for m in ("orig", "cbs") + MODEL_IDS:
                    if f"{m}_delay" not in s:
                        continue
                    row[f"{tag}_{m}_delay"] = round(s.get(f"{m}_delay", float("nan")), 2)
                    row[f"{tag}_{m}_makespan"] = round(s.get(f"{m}_makespan", float("nan")), 2)
                    if m not in ("orig", "cbs"):
                        row[f"{tag}_{m}_gap"] = round(s.get(f"{m}_gap", float("nan")), 1)
                row[f"{tag}_expert_delay"] = round(s["expert_delay"], 2)
                row[f"{tag}_expert_makespan"] = round(s.get("expert_makespan", float("nan")), 2)
            row["minutes"] = round((time.time() - t0) / 60, 1)
            rows.append(row)
            _write_summary(root, rows, tags, ename)  # updated after every experiment and map

    plot_sweep_summary(rows, tags, os.path.join(root, "sweep_summary.png"),
                       expert_name="CBS (expert)" if ename == "CBS" else ename)
    log(f"\n[sweep] done. {os.path.join(root, 'sweep_summary.md')}")
    with open(os.path.join(root, "sweep_summary.md"), encoding="utf-8") as f:
        log(f.read())
    cmd = scp_command(cfg, root)
    if cmd:
        log("[sweep] to copy the results to your PC, run this on the PC:\n\n" + cmd + "\n")
    return root, rows


def _server_address():
    """This machine's address as seen from the PC: the one the SSH session came in on
    (SSH_CONNECTION), else the address of the default network interface."""
    import socket

    ssh = os.environ.get("SSH_CONNECTION", "").split()
    if len(ssh) == 4:
        return ssh[2]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


def scp_command(cfg, path):
    """``scp -r user@host:/abs/path <scp_dest>`` to copy ``path`` from this machine to the
    PC, or None (on Windows, where the files are already local, or when scp_dest is off)."""
    import getpass

    dest = cfg.get("scp_dest", "C:\\Users\\Alpha\\Downloads\\")
    if not dest or os.name == "nt":
        return None
    user = cfg.get("scp_user") or getpass.getuser()
    host = cfg.get("scp_host") or _server_address()
    if ":" in host:  # IPv6
        host = f"[{host}]"
    src = os.path.abspath(path).replace(os.sep, "/")
    if " " in src:
        src = f'"{src}"'
    return f"scp -r {user}@{host}:{src} {dest}"


def _best_epochs(cfg, grid, n, ids):
    import torch

    from mapf_nco.paths import find_checkpoint

    out = {}
    for kind in ids:
        path = find_checkpoint(cfg, grid.name, n, kind)
        if path is None:
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        out[f"{kind}_best_epoch"] = ck.get("epoch")
        out[f"{kind}_val_delay"] = round(float(ck.get("val_delay", float("nan"))), 2)
    return out


TAG_NAMES = {"test": "(test set)", "second": "(second test set)"}


def _write_summary(root, rows, tags, ename):
    if ename == "CBS":  # Python CBS as the expert: don't confuse it with the original-CBS column
        ename = "CBS (expert)"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(root, "sweep_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    settings = ["map", "agents"] + (["second_agents"] if "second_agents" in keys else []) \
        + (["scaling"] if len({r.get("scaling") for r in rows}) > 1 else []) \
        + [k for k in keys if k in SWEEPABLE]
    ids = [m for m in MODEL_IDS
           if any(f"{m}_best_epoch" in r for r in rows)]
    names = {m: f"{planner_of(m).upper()} {KIND_LABELS[kind_of(m)]}" for m in MODEL_IDS}
    pp = [m for m in ids if planner_of(m) == "pp"]
    cbs = [m for m in ids if planner_of(m) == "cbs"]
    groups = []  # (column title, per-tag keys)
    if pp:
        groups.append(("PP / " + " / ".join(names[m] for m in pp) + f" / {ename}",
                       ["orig"] + pp + ["expert"]))
    if cbs:
        tail = [] if pp else ["expert"]
        groups.append(("CBS / " + " / ".join(names[m] for m in cbs)
                       + ("" if pp else f" / {ename}"), ["cbs"] + cbs + tail))
    if not groups:
        groups.append((f"PP / {ename}", ["orig", "expert"]))
    head = (["exp"] + settings + ["best ep. " + " / ".join(names[m] for m in ids),
                                  "val delay " + " / ".join(names[m] for m in ids)]
            + [f"delay {TAG_NAMES[t]}: {g}" for t in tags for g, _ in groups]
            + [f"makespan {TAG_NAMES[t]}: {g}" for t in tags for g, _ in groups] + ["min"])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["experiment"][4:]] + [str(r.get(k, "-")) for k in settings]
        cells.append(" / ".join(str(r.get(f"{m}_best_epoch", "-")) for m in ids))
        cells.append(" / ".join(str(r.get(f"{m}_val_delay", "-")) for m in ids))
        for what in ("delay", "makespan"):
            for tag in tags:
                for _, cols in groups:
                    cells.append(" / ".join(str(r.get(f"{tag}_{m}_{what}", "-")) for m in cols))
        cells.append(str(r["minutes"]))
        lines.append("| " + " | ".join(cells) + " |")
    notes = [f"- exp {r['experiment'][4:]}: {r['not_trained']}" for r in rows
             if r.get("not_trained")]
    if notes:
        lines += ["", "Not trained (the original CBS timed out on most validation scenarios at "
                  "this agent count, see the log; '-' in the table):"] + notes
    with open(os.path.join(root, "sweep_summary.md"), "w", encoding="utf-8") as f:
        targets = ", ".join(f"{m} ({a} agents)"
                            for m, a in dict.fromkeys((r["map"], r["agents"]) for r in rows))
        sets = " and ".join(TAG_NAMES[t].strip("()") for t in tags)
        pos = ("Start/goal positions: **fixed** (one scenario per map and agent count, used for "
               "training, validation and test).\n\n" if rows[0].get("positions") == "fixed"
               else "Start/goal positions: randomized (new for every instance).\n\n")
        f.write(f"# Sweep: {targets}\n\n{pos}Average delay and makespan on the {sets} of each map. "
                f"PP columns: original PP / learned PP / {ename}; "
                f"CBS columns: original CBS / CBS with learned costs. Lower is better.\n\n"
                + "\n".join(lines) + "\n")
