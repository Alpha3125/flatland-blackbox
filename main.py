"""MAPF NCO: replication of "Learned Representations Enhance Multi Agent Path
Planning" (Captari & van Hoof, ICML 2025 PRAL workshop).

Run without arguments for the interactive menu:

    python main.py                              (uses configs/paper.json)

    1) Generate  scenarios with pre-computed EECBS (expert), CBS (optimal) and PP plans
    2) Experiment  train the Edge / Potential / Potential+Edge / Collision model for PP, CBS or both
    3) Test  on the saved test scenarios, or on new random ones with any number of agents

Or use the commands directly:

    python main.py generate   --agents 50
    python main.py experiment --agents 50 --epochs 30 --lr 5e-4 --lam 1.0
    python main.py experiment --agents 20 --planner cbs                   CBS models only
    python main.py experiment --agents 50 --model potential_edge --planner pp
    python main.py test       --agents 50 --num 10                       saved test scenarios
    python main.py test       --agents 50 --random 5 --test_agents 75     new random scenarios
    python main.py sweep      --agents 50 --grid lr=1e-4,5e-4 lam=0.5,1   4 experiments
    python main.py sweep      --agents 50 --grid model=edge,potential,potential_edge
    python main.py evaluate   --agents 50 75 100                          Table 1, no plots
    python main.py clean      --agents 20      (or --what sweeps / --what all)

``--set key=value`` overrides any entry of the config file.
"""

import argparse
import json
import os
import sys

DEFAULT_CONFIG = "configs/paper.json"
# Model ids accepted by test / evaluate --model (kept here so parsing needs no imports).
MODEL_ID_CHOICES = ("edge", "potential", "potential_edge", "collision",
                    "edge_cbs", "potential_cbs", "potential_edge_cbs", "collision_cbs")


# ------------------------------------------------------------------ config
def load_config(path, overrides=None, **explicit):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for item in overrides or []:
        key, _, value = item.partition("=")
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value
    for key, v in explicit.items():
        if v is not None:
            cfg[key] = v
    return cfg


def default_workers():
    return max(1, (os.cpu_count() or 2) - 1)


# --------------------------------------------------------- interactive menu
def ask(prompt, default=None, cast=str, choices=None):
    """Ask until a valid answer is given. Empty input returns the default."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            val = cast(raw)
        except ValueError:
            print("  please enter a valid value")
            continue
        if choices is not None and val not in choices:
            print(f"  choose one of: {', '.join(map(str, choices))}")
            continue
        return val


def ask_yes(prompt, default=True):
    d = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{d}]: ").strip().lower()
    return default if not raw else raw.startswith("y")


def open_file(path):
    """Open a picture with the system viewer (best effort)."""
    try:
        if sys.platform == "win32":
            os.startfile(os.path.abspath(path))  # noqa: S606
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}" >/dev/null 2>&1 &')
    except Exception:
        pass


def status(cfg, n):
    from mapf_nco.grid import Grid
    from mapf_nco.paths import checkpoint_path, run_dir
    from mapf_nco.scenarios import dataset_path, load_dataset

    grid = Grid.from_file(cfg["map_path"])
    info = {"grid": grid}
    print(f"\n--- {grid.name}, {n} agents ---")
    for split in ("train", "test"):
        p = dataset_path(cfg["data_dir"], grid.name, n, split)
        if os.path.exists(p):
            scs, meta = load_dataset(p, grid)
            info[split] = len(scs)
            pp = sum(s.pp_paths is not None for s in scs)
            cbs = sum(s.cbs_paths is not None for s in scs)
            info[f"{split}_missing"] = pp < len(scs) or (
                cbs < len(scs) and not all(s.meta.get("cbs_timed_out") or s.cbs_paths
                                           for s in scs))
            print(f"  {split:5s} data : {len(scs)} scenarios; plans: {meta.get('expert')} "
                  f"{len(scs)}, PP {pp}, CBS {cbs}")
        else:
            info[split] = 0
            info[f"{split}_missing"] = False
            print(f"  {split:5s} data : not generated yet")
    from mapf_nco.planners import LABELS, MODEL_IDS

    for mid in MODEL_IDS:
        info[mid] = os.path.exists(checkpoint_path(cfg, grid.name, n, mid))
    trained = [LABELS[m] for m in MODEL_IDS if info[m]]
    print(f"  models     : {', '.join(trained) if trained else 'none trained yet'}"
          + (f"  ({os.path.dirname(checkpoint_path(cfg, grid.name, n, 'edge'))})" if trained else ""))
    info["any_model"] = bool(trained)
    print(f"  outputs    : {run_dir(cfg, grid.name, n)}")
    return info


def _expert_available(cfg):
    from mapf_nco.solver_eecbs import explain_missing, find_eecbs

    if cfg.get("expert", "eecbs") != "eecbs" or find_eecbs(cfg.get("eecbs_bin")) is not None:
        return True
    print("  " + explain_missing(cfg.get("eecbs_bin")).replace("\n", "\n  "))
    if not ask_yes("  Use the slow Python CBS instead (only OK for ~20 agents or fewer)?", False):
        return False
    cfg["expert"] = "cbs"
    return True


def menu_generate(cfg, n, info):
    from mapf_nco.experts import generate_datasets

    cfg = dict(cfg)
    if info["train"] or info["test"]:
        c = ask("Data for this agent count exists: 1) add missing PP/CBS plans to it  "
                "2) replace it  0) cancel", 1 if info["train_missing"] or info["test_missing"]
                else 0, int, [0, 1, 2])
        if c == 0:
            return
        cfg["augment" if c == 1 else "overwrite"] = True
    if not cfg.get("augment"):
        cfg["num_train"] = ask("Number of training scenarios (part goes to validation)",
                               cfg["num_train"], int)
        cfg["num_test"] = ask("Number of test scenarios", cfg["num_test"], int)
    cfg["with_cbs"] = ask_yes("Also pre-compute optimal CBS plans (slow for many agents)?",
                              cfg.get("with_cbs", True))
    if cfg["with_cbs"]:
        cfg["cbs_time_limit"] = ask("  CBS time limit per scenario, seconds",
                                    cfg.get("cbs_time_limit", 60), float)
    if not _expert_available(cfg):
        return
    generate_datasets(cfg, [n], workers=default_workers())


def ask_hyperparameters(cfg):
    """Every training setting, with the config values as defaults."""
    cfg = dict(cfg)
    print("  Training  [Enter keeps the default]")
    print("    model: edge           = one learned cost per edge (the paper's Edge model)")
    print("           potential      = PotP: learned strengths for moving towards obstructions,")
    print("                            into single-lane passages and dead ends, plus one-way lanes")
    print("           potential_edge = potential first, then the Edge model starting from its costs")
    print("           collision      = per scenario: dearer into cells where collisions happen,")
    print("                            cheaper into clean cells (4 learned strengths)")
    cfg["model"] = ask("    model", cfg.get("model", "edge"), str,
                       ["edge", "potential", "potential_edge", "collision"])
    if cfg["model"] == "collision":
        init = dict(cfg.get("collision_init", {}))
        print("    collision model's starting strengths (0 = no effect; positive = dearer / "
              "cheaper for clean):")
        for key, text in (("naive", "into a cell where individual shortest paths collide"),
                          ("gave_way", "into a cell where the original PP plan waited"),
                          ("close_call", "into a cell with a close call in the original PP"),
                          ("clean", "cheaper: into a cell with none of these")):
            init[key] = ask(f"      {key:10s} ({text})", float(init.get(key, 0.0)), float)
        cfg["collision_init"] = init
    elif cfg["model"] != "edge":
        init = dict(cfg.get("potential_init", {}))
        print("    potential model's starting strengths (0 = no effect; positive = dearer):")
        for key, text in (("toward", "moving towards an obstruction"),
                          ("corridor", "moving into a single-lane passage"),
                          ("near", "being next to an obstruction"),
                          ("dead_end", "moving into a dead end"),
                          ("lanes", "alternating one-way rows/columns")):
            init[key] = ask(f"      {key:9s} ({text})", float(init.get(key, 0.0)), float)
        cfg["potential_init"] = init
    print("    planner: pp   = learn edge costs for Prioritized Planning (the paper)")
    print("             cbs  = learn edge costs for CBS (optimal; slow for many agents)")
    print("             both = a separate model for each")
    cfg["planner"] = ask("    planner", cfg.get("planner", "both"), str, ["pp", "cbs", "both"])
    if cfg["planner"] != "pp":
        cfg["cbs_train_time_limit"] = ask("    CBS time limit per call during training, seconds",
                                          cfg.get("cbs_train_time_limit", 10), float)
    cfg["epochs"] = ask("    epochs", cfg["epochs"], int)
    cfg["batch_size"] = ask("    batch size (scenarios per update)", cfg["batch_size"], int)
    cfg["lr"] = ask("    learning rate (Adam)", cfg["lr"], float)
    cfg["lam"] = ask("    lambda (perturbation strength, Eq. 5)", cfg["lam"], float)
    cfg["objective"] = ask("    objective: usage (paper: match EECBS), makespan (minimise the "
                           "latest arrival), usage+makespan", cfg.get("objective", "usage"), str,
                           ["usage", "makespan", "usage+makespan"])
    if cfg["objective"] != "usage":
        cfg["makespan_tau"] = ask("    makespan smoothness tau (small = closer to the true max)",
                                  cfg.get("makespan_tau", 1.0), float)
    if cfg["objective"] == "usage+makespan":
        cfg["makespan_weight"] = ask("    weight of the makespan term",
                                     cfg.get("makespan_weight", 0.001), float)
    if cfg["objective"] != "makespan":
        cfg["loss_reduction"] = ask("    usage loss: mean (paper, 1/M) or sum",
                                    cfg["loss_reduction"], str, ["mean", "sum"])
    cfg["w_min"] = ask("    w_min (smallest allowed edge cost)", cfg["w_min"], float)
    cfg["val_fraction"] = ask("    validation fraction of the training scenarios",
                              cfg["val_fraction"], float)
    cfg["seed"] = ask("    random seed", cfg["seed"], int)
    cfg["workers"] = ask("  parallel workers", default_workers(), int)
    return cfg


def menu_experiment(cfg, n, info):
    from mapf_nco.paths import training_dir
    from mapf_nco.planners import training_runs
    from mapf_nco.train import train

    if not info["train"]:
        print("  No training data yet: generate first (option 1).")
        return
    cfg = ask_hyperparameters(cfg)
    for kind, planner in training_runs(cfg):
        print()
        train(cfg, n, planner, workers=cfg["workers"], log=print, kind=kind)
    curves = os.path.join(training_dir(cfg, info["grid"].name, n), "curves.png")
    print(f"\nExperiment finished. Loss graph (all models): {curves}")
    open_file(curves)
    print("Use option 3 to test the models.")


def menu_test(cfg, n, info):
    from mapf_nco.evaluate import test_random, test_saved
    from mapf_nco.paths import test_dir

    if not info["any_model"]:
        print("  (no trained models yet: the test compares original PP with EECBS/CBS only)")
    c = ask("Test on: 1) the saved test scenarios  2) new random scenarios", 1 if info["test"]
            else 2, int, [1, 2])
    if c == 1:
        if not info["test"]:
            print("  No saved test scenarios: generate first (option 1).")
            return
        k = ask(f"  How many test instances (1-{info['test']}; more = steadier average)",
                min(3, info["test"]), int,
                list(range(1, info["test"] + 1)))
        animate = ask_yes("  Also save animations (GIF, slower)?", False)
        test_saved(cfg, n, k, plots=True, animate=animate)
    else:
        cfg = dict(cfg)
        agents = ask(f"  Number of agents (models were trained on {n})", n, int)
        count = ask("  How many random scenarios (more = steadier average)", 3, int)
        seed = ask("  Random seed", 12345, int)
        cfg["with_cbs"] = ask_yes("  Also solve with optimal CBS?", cfg.get("with_cbs", True))
        animate = ask_yes("  Also save animations (GIF, slower)?", False)
        if not _expert_available(cfg):
            return
        test_random(cfg, n, agents, count, seed, plots=True, animate=animate,
                    workers=default_workers())
    open_file(os.path.join(test_dir(cfg, info["grid"].name, n), "summary.png"))


def _ask_values(prompt, default, cast):
    """Comma-separated values, e.g. '1e-4, 5e-4'."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return [default]
        try:
            vals = [cast(v.strip()) for v in raw.split(",") if v.strip()]
            if vals:
                return vals
        except ValueError:
            pass
        print("  enter one value, or several separated by commas")


def menu_sweep(cfg, n, info):
    from mapf_nco.experts import generate_datasets
    from mapf_nco.sweep import expand_grid, run_sweep, second_agent_count

    cfg = dict(cfg)
    print("  Sweep = generate (if needed) + a queue of experiments, each trained and tested on the")
    print("  same scenarios. Give several values separated by commas to try them all.")
    if not _expert_available(cfg):
        return
    if not (info["train"] and info["test"]):
        print("  No data for this agent count yet: generating it first.")
        cfg["num_train"] = ask("  Number of training scenarios", cfg["num_train"], int)
        cfg["num_test"] = ask("  Number of test scenarios", cfg["num_test"], int)
        need_data = True
    else:
        need_data = info["train_missing"] or info["test_missing"]
        if need_data:
            print("  (existing data will get its missing PP/CBS plans added)")
            cfg["augment"] = True
    grid_values = {
        "model": _ask_values("  model (edge / potential / potential_edge / collision)",
                             cfg.get("model", "edge"), str),
        "planner": _ask_values("  planner (pp / cbs / both)", cfg.get("planner", "both"), str),
        "objective": _ask_values("  objective (usage / makespan / usage+makespan)",
                                 cfg.get("objective", "usage"), str),
        "makespan_tau": _ask_values("  makespan tau (used by the makespan objectives)",
                                    cfg.get("makespan_tau", 1.0), float),
        "makespan_weight": _ask_values("  makespan weight (used by usage+makespan)",
                                       cfg.get("makespan_weight", 0.001), float),
        "epochs": _ask_values("  epochs", cfg["epochs"], int),
        "batch_size": _ask_values("  batch size", cfg["batch_size"], int),
        "lr": _ask_values("  learning rate", cfg["lr"], float),
        "lam": _ask_values("  lambda", cfg["lam"], float),
        "loss_reduction": _ask_values("  loss (mean/sum)", cfg["loss_reduction"], str),
        "w_min": _ask_values("  w_min", cfg["w_min"], float),
        "val_fraction": _ask_values("  validation fraction", cfg["val_fraction"], float),
        "seed": _ask_values("  seed", cfg["seed"], int),
    }
    if any(v not in ("edge", "potential", "potential_edge", "collision")
           for v in grid_values["model"]):
        print("  model must be edge, potential, potential_edge or collision")
        return
    if any(v not in ("pp", "cbs", "both") for v in grid_values["planner"]):
        print("  planner must be pp, cbs or both")
        return
    if any(v not in ("mean", "sum") for v in grid_values["loss_reduction"]):
        print("  loss must be 'mean' or 'sum'")
        return
    if any(v not in ("usage", "makespan", "usage+makespan") for v in grid_values["objective"]):
        print("  objective must be usage, makespan or usage+makespan")
        return
    k = len(expand_grid(grid_values))
    test_count = ask("  Test scenarios per experiment (from the saved test set)",
                     min(3, info["test"] or cfg["num_test"]), int)
    second_n = second_count = None
    if ask_yes("  Also test on a second set with a different number of agents?", False):
        second_n = ask("  Agents in the second test set", second_agent_count(n, cfg), int)
        second_count = ask("  Scenarios in the second test set", test_count, int)
    animate = ask_yes("  Save animations too (slower)?", False)
    workers = ask("  Parallel workers", default_workers(), int)
    print(f"\n  {k} experiment(s) x (train the model(s) per planner + {2 if second_n else 1} "
          f"test(s)). "
          f"Each training takes about epochs x (a few seconds to a minute).")
    if not ask_yes("  Start?", True):
        return
    if need_data:
        generate_datasets(cfg, [n], workers=workers)
    root, _ = run_sweep(cfg, n, grid_values, test_count, second_n, second_count,
                        animate=animate, workers=workers)
    open_file(os.path.join(root, "sweep_summary.png"))


def clear_outputs(cfg, what, map_name=None, n=None):
    """Delete outputs. what: 'agents' (one agent count), 'sweeps' or 'all'. Never touches data/."""
    import shutil

    from mapf_nco.paths import run_dir

    out = os.path.abspath(cfg["output_dir"])
    target = {"agents": run_dir(cfg, map_name, n) if n is not None else None,
              "sweeps": os.path.join(cfg["output_dir"], "sweeps"),
              "all": cfg["output_dir"]}[what]
    target = os.path.abspath(target)
    if not (target == out or target.startswith(out + os.sep)):
        raise ValueError(f"refusing to delete {target}: outside {out}")
    if not os.path.isdir(target):
        print(f"  nothing to clear at {target}")
        return
    if target == out:  # keep the folder itself (and .gitkeep)
        for name in os.listdir(out):
            if name == ".gitkeep":
                continue
            p = os.path.join(out, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    else:
        shutil.rmtree(target)
    print(f"  cleared {target}")


def menu_clear(cfg, n, info):
    from mapf_nco.paths import run_dir

    print(f"  1) results for {n} agents   ({run_dir(cfg, info['grid'].name, n)})")
    print(f"  2) all sweeps               ({os.path.join(cfg['output_dir'], 'sweeps')})")
    print(f"  3) everything in outputs    ({cfg['output_dir']})")
    print("     (the generated scenarios in data/ are never deleted)")
    c = ask("  Clear", 0, int, [0, 1, 2, 3])
    if c == 0:
        return
    what = {1: "agents", 2: "sweeps", 3: "all"}[c]
    if ask_yes("  This deletes trained models and results permanently. Sure?", False):
        clear_outputs(cfg, what, info["grid"].name, n)


def interactive(config_path):
    cfg = load_config(config_path)
    print("=" * 64)
    print(" MAPF NCO: learned edge costs for Prioritized Planning and CBS")
    print(f" config: {config_path}")
    print("=" * 64)
    n = ask("Number of agents (for data and models)", cfg["agent_counts"][0], int)
    while True:
        info = status(cfg, n)
        print("""
  1) Generate scenarios + EECBS, CBS and PP plans
  2) Experiment: train a model (Edge / Potential / Potential+Edge / Collision; choose settings)
  3) Test: saved test scenarios, or new random scenarios (comparison plots)
  4) Change number of agents
  5) Sweep: generate + queue of experiments + tests
  6) Clear outputs (models, plots, results; generated data is kept)
  0) Quit""")
        c = ask("Choose", None, int, [0, 1, 2, 3, 4, 5, 6])
        try:
            if c == 0:
                return
            if c == 1:
                menu_generate(cfg, n, info)
            elif c == 2:
                menu_experiment(cfg, n, info)
            elif c == 3:
                menu_test(cfg, n, info)
            elif c == 4:
                n = ask("Number of agents", n, int)
            elif c == 5:
                menu_sweep(cfg, n, info)
            elif c == 6:
                menu_clear(cfg, n, info)
        except KeyboardInterrupt:
            print("\n  (interrupted, back to menu)")
        except Exception as e:  # keep the menu alive, show what went wrong
            import traceback

            traceback.print_exc()
            print(f"\n  Error: {e}")


# --------------------------------------------------------------- commands
HPARAMS = ("epochs", "batch_size", "lr", "lam", "loss_reduction", "w_min", "val_fraction",
           "seed", "device", "expert", "eecbs_bin", "cbs_time_limit", "objective",
           "makespan_tau", "makespan_weight", "planner", "model")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", dest="menu_config", default=DEFAULT_CONFIG,
                   help="config for the interactive menu")
    sub = p.add_subparsers(dest="command")

    def common(sp):
        sp.add_argument("--config", default=DEFAULT_CONFIG)
        sp.add_argument("--set", nargs="*", metavar="KEY=VALUE",
                        help="override config entries, e.g. lr=0.001 'suboptimality={\"20\": 1.0}'")
        sp.add_argument("--agents", type=int, nargs="+", help="agent counts (default: config)")
        sp.add_argument("--workers", type=int, default=None,
                        help="parallel processes (default: CPU count - 1)")

    g = sub.add_parser("generate", help="scenarios + EECBS, CBS and PP plans")
    common(g)
    g.add_argument("--expert", choices=["eecbs", "cbs"])
    g.add_argument("--eecbs_bin")
    g.add_argument("--cbs_time_limit", type=float)
    g.add_argument("--no_cbs", action="store_true", help="skip the optimal CBS plans")
    g.add_argument("--overwrite", action="store_true", help="replace existing data")
    g.add_argument("--augment", action="store_true",
                   help="keep existing data, add missing PP/CBS plans")

    x = sub.add_parser("experiment",
                       help="train the Edge / Potential / Potential+Edge / Collision model")
    common(x)
    x.add_argument("--model", choices=["edge", "potential", "potential_edge", "collision"],
                   help="how the edge costs are learned (default: config, edge); set starting "
                        "strengths with --set potential_init='{\"toward\": 0.2}' or "
                        "collision_init='{\"naive\": 0.05}'")
    x.add_argument("--planner", choices=["pp", "cbs", "both"],
                   help="downstream planner the costs are learned for (default: config, both)")
    for name, typ in (("epochs", int), ("batch_size", int), ("lr", float), ("lam", float),
                      ("w_min", float), ("val_fraction", float), ("seed", int)):
        x.add_argument(f"--{name}", type=typ)
    x.add_argument("--loss_reduction", choices=["mean", "sum"])
    x.add_argument("--objective", choices=["usage", "makespan", "usage+makespan"],
                   help="usage = paper (match EECBS); makespan = minimise the latest arrival")
    x.add_argument("--makespan_tau", type=float)
    x.add_argument("--makespan_weight", type=float)
    x.add_argument("--device")

    t = sub.add_parser("test", help="compare PP / EECBS / CBS / trained PP with plots")
    common(t)
    t.add_argument("--model", dest="models", nargs="+", choices=list(MODEL_ID_CHOICES),
                   help="models to test, e.g. edge potential_cbs (default: every trained model)")
    t.add_argument("--num", type=int, help="saved test instances (default: all)")
    t.add_argument("--random", type=int, metavar="K",
                   help="test on K new random scenarios instead of the saved ones")
    t.add_argument("--test_agents", type=int, help="agents in the random scenarios")
    t.add_argument("--seed", type=int, default=12345, help="seed of the random scenarios")
    t.add_argument("--no_cbs", action="store_true")
    t.add_argument("--animate", action="store_true")
    t.add_argument("--expert", choices=["eecbs", "cbs"])
    t.add_argument("--eecbs_bin")

    w = sub.add_parser("sweep", help="queue of experiments, each tested on 2 agent counts")
    common(w)
    w.add_argument("--grid", nargs="+", metavar="KEY=V1,V2", default=[],
                   help="values to try, e.g. lr=1e-4,5e-4 lam=0.5,1 planner=pp,cbs")
    w.add_argument("--num", type=int, help="saved test instances per experiment")
    w.add_argument("--second_agents", type=int, help="agents in the second test set")
    w.add_argument("--second_num", type=int, help="scenarios in the second test set")
    w.add_argument("--animate", action="store_true")
    w.add_argument("--expert", choices=["eecbs", "cbs"])
    w.add_argument("--eecbs_bin")

    c = sub.add_parser("clean", help="delete outputs (models, plots, results); data/ is kept")
    common(c)
    c.add_argument("--what", choices=["agents", "sweeps", "all"], default="agents",
                   help="agents = outputs of --agents (default); sweeps; all = all of outputs/")
    c.add_argument("--yes", action="store_true", help="do not ask for confirmation")

    e = sub.add_parser("evaluate", help="Table 1 over all saved test scenarios, no plots")
    common(e)
    e.add_argument("--model", dest="models", nargs="+", choices=list(MODEL_ID_CHOICES))
    return p.parse_args(argv)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # so "Σ" prints on any Windows console
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    if args.command is None:
        return interactive(args.menu_config)

    explicit = {k: getattr(args, k, None) for k in HPARAMS}
    if args.command in ("test",):
        explicit.pop("seed")  # the test's --seed is the scenario seed, not training
    for flag in ("overwrite", "augment"):
        if getattr(args, flag, False):
            explicit[flag] = True
    if getattr(args, "no_cbs", False):
        explicit["with_cbs"] = False
    cfg = load_config(args.config, args.set, **explicit)
    agents = args.agents or cfg["agent_counts"]
    from mapf_nco.planners import MODEL_IDS, training_runs

    models = getattr(args, "models", None)
    workers = args.workers or default_workers()

    if args.command == "generate":
        from mapf_nco.experts import generate_datasets

        generate_datasets(cfg, agents, workers=workers)

    if args.command == "experiment":
        from mapf_nco.train import train

        for n in agents:
            for kind, planner in training_runs(cfg):
                train(cfg, n, planner, workers=workers, kind=kind)

    if args.command == "test":
        from mapf_nco.evaluate import test_random, test_saved

        for n in agents:
            if args.random:
                test_random(cfg, n, args.test_agents or n, args.random, args.seed,
                            models or MODEL_IDS,
                            animate=args.animate, workers=workers)
            else:
                test_saved(cfg, n, args.num, models or MODEL_IDS, animate=args.animate)

    if args.command == "sweep":
        from mapf_nco.experts import generate_datasets
        from mapf_nco.sweep import SWEEPABLE, run_sweep

        grid_values = {}
        for item in args.grid:
            key, _, vals = item.partition("=")
            if key not in SWEEPABLE:
                raise SystemExit(f"cannot sweep '{key}'; choose from {', '.join(SWEEPABLE)}")
            words = ("pp", "cbs", "both", "mean", "sum", "usage", "makespan", "usage+makespan",
                     "edge", "potential", "potential_edge", "collision")
            grid_values[key] = [v if v in words else json.loads(v) for v in vals.split(",")]
        for n in agents:
            generate_datasets(dict(cfg, augment=True), [n], workers=workers)
            run_sweep(cfg, n, grid_values or {"lr": [cfg["lr"]]}, args.num, args.second_agents,
                      args.second_num, animate=args.animate, workers=workers)

    if args.command == "clean":
        from mapf_nco.grid import Grid

        name = Grid.from_file(cfg["map_path"]).name
        if args.yes or ask_yes(f"Delete outputs ({args.what})? Trained models are lost.", False):
            for n in (agents if args.what == "agents" else [None]):
                clear_outputs(cfg, args.what, name, n)

    if args.command == "evaluate":
        from mapf_nco.evaluate import evaluate, format_table, save_results

        rows = evaluate(cfg, agents, models or MODEL_IDS)
        print("\n" + format_table(rows))
        csv_path, md_path = save_results(rows, os.path.join(cfg["output_dir"], "results"))
        print(f"\nSaved {csv_path} and {md_path}")


if __name__ == "__main__":
    sys.exit(main())
