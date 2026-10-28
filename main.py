"""MAPF NCO: replication of "Learned Representations Enhance Multi Agent Path
Planning" (Captari & van Hoof, ICML 2025 PRAL workshop).

Run without arguments for the interactive menu:

    python main.py                              (uses configs/paper.json)

    1) Generate  scenarios with pre-computed EECBS (expert), CBS (optimal) and PP plans
    2) Experiment  train both the Edge and the GNN model (asks every hyperparameter)
    3) Test  on the saved test scenarios, or on new random ones with any number of agents

Or use the commands directly:

    python main.py generate   --agents 50
    python main.py experiment --agents 50 --epochs 30 --lr 5e-4 --lam 1.0
    python main.py test       --agents 50 --num 10                       saved test scenarios
    python main.py test       --agents 50 --random 5 --test_agents 75     new random scenarios
    python main.py evaluate   --agents 50 75 100                          Table 1, no plots

``--set key=value`` overrides any entry of the config file.
"""

import argparse
import json
import os
import sys

DEFAULT_CONFIG = "configs/paper.json"


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
    for kind in ("edge", "gnn"):
        ck = checkpoint_path(cfg, grid.name, n, kind)
        info[kind] = os.path.exists(ck)
        print(f"  {kind:5s} model: {'trained  ' + ck if info[kind] else 'not trained yet'}")
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
    g = dict(cfg.get("gnn", {}))
    print("  Training (both models)  [Enter keeps the default]")
    cfg["epochs"] = ask("    epochs", cfg["epochs"], int)
    cfg["batch_size"] = ask("    batch size (scenarios per update)", cfg["batch_size"], int)
    cfg["lr"] = ask("    learning rate (Adam)", cfg["lr"], float)
    cfg["lam"] = ask("    lambda (perturbation strength, Eq. 5)", cfg["lam"], float)
    cfg["loss_reduction"] = ask("    loss: mean (paper, 1/M) or sum", cfg["loss_reduction"], str,
                                ["mean", "sum"])
    cfg["w_min"] = ask("    w_min (smallest allowed edge cost)", cfg["w_min"], float)
    cfg["val_fraction"] = ask("    validation fraction of the training scenarios",
                              cfg["val_fraction"], float)
    cfg["seed"] = ask("    random seed", cfg["seed"], int)
    print("  GNN architecture")
    g["hidden"] = ask("    hidden size (GraphSAGE)", g.get("hidden", 64), int)
    g["num_freqs"] = ask("    positional-embedding frequencies", g.get("num_freqs", 6), int)
    g["id_dim"] = ask("    agent-ID embedding size", g.get("id_dim", 8), int)
    g["mlp_hidden"] = ask("    edge-MLP hidden size", g.get("mlp_hidden", 64), int)
    cfg["gnn"] = g
    cfg["workers"] = ask("  parallel workers", default_workers(), int)
    return cfg


def menu_experiment(cfg, n, info):
    from mapf_nco.paths import training_dir
    from mapf_nco.train import train

    if not info["train"]:
        print("  No training data yet: generate first (option 1).")
        return
    cfg = ask_hyperparameters(cfg)
    results = {}
    for kind in ("edge", "gnn"):
        print()
        train(cfg, n, kind, workers=cfg["workers"], log=print)
        results[kind] = os.path.join(training_dir(cfg, info["grid"].name, n, kind), "curves.png")
    print("\nExperiment finished. Loss graphs:")
    for kind, path in results.items():
        print(f"  {kind}: {path}")
        open_file(path)
    print("Use option 3 to test the models.")


def menu_test(cfg, n, info):
    from mapf_nco.evaluate import test_random, test_saved
    from mapf_nco.paths import test_dir

    if not (info["edge"] or info["gnn"]):
        print("  (no trained models yet: the test compares original PP with EECBS/CBS only)")
    c = ask("Test on: 1) the saved test scenarios  2) new random scenarios", 1 if info["test"]
            else 2, int, [1, 2])
    if c == 1:
        if not info["test"]:
            print("  No saved test scenarios: generate first (option 1).")
            return
        k = ask(f"  How many test instances (1-{info['test']})", info["test"], int,
                list(range(1, info["test"] + 1)))
        animate = ask_yes("  Also save animations (GIF, slower)?", False)
        test_saved(cfg, n, k, plots=True, animate=animate)
    else:
        cfg = dict(cfg)
        agents = ask(f"  Number of agents (models were trained on {n})", n, int)
        count = ask("  How many random scenarios", 5, int)
        seed = ask("  Random seed", 12345, int)
        cfg["with_cbs"] = ask_yes("  Also solve with optimal CBS?", cfg.get("with_cbs", True))
        animate = ask_yes("  Also save animations (GIF, slower)?", False)
        if not _expert_available(cfg):
            return
        test_random(cfg, n, agents, count, seed, plots=True, animate=animate,
                    workers=default_workers())
    open_file(os.path.join(test_dir(cfg, info["grid"].name, n), "summary.png"))


def interactive(config_path):
    cfg = load_config(config_path)
    print("=" * 64)
    print(" MAPF NCO: learned edge costs for Prioritized Planning")
    print(f" config: {config_path}")
    print("=" * 64)
    n = ask("Number of agents (for data and models)", cfg["agent_counts"][0], int)
    while True:
        info = status(cfg, n)
        print("""
  1) Generate scenarios + EECBS, CBS and PP plans
  2) Experiment: train the Edge and GNN models (choose hyperparameters)
  3) Test: saved test scenarios, or new random scenarios (comparison plots)
  4) Change number of agents
  0) Quit""")
        c = ask("Choose", None, int, [0, 1, 2, 3, 4])
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
        except KeyboardInterrupt:
            print("\n  (interrupted, back to menu)")
        except Exception as e:  # keep the menu alive, show what went wrong
            import traceback

            traceback.print_exc()
            print(f"\n  Error: {e}")


# --------------------------------------------------------------- commands
HPARAMS = ("epochs", "batch_size", "lr", "lam", "loss_reduction", "w_min", "val_fraction",
           "seed", "device", "expert", "eecbs_bin", "cbs_time_limit")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", dest="menu_config", default=DEFAULT_CONFIG,
                   help="config for the interactive menu")
    sub = p.add_subparsers(dest="command")

    def common(sp):
        sp.add_argument("--config", default=DEFAULT_CONFIG)
        sp.add_argument("--set", nargs="*", metavar="KEY=VALUE",
                        help="override config entries, e.g. gnn='{\"hidden\": 32}'")
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

    x = sub.add_parser("experiment", help="train both the Edge and the GNN model")
    common(x)
    x.add_argument("--model", nargs="+", choices=["edge", "gnn"], help="default: both")
    for name, typ in (("epochs", int), ("batch_size", int), ("lr", float), ("lam", float),
                      ("w_min", float), ("val_fraction", float), ("seed", int)):
        x.add_argument(f"--{name}", type=typ)
    x.add_argument("--loss_reduction", choices=["mean", "sum"])
    x.add_argument("--device")

    t = sub.add_parser("test", help="compare PP / EECBS / CBS / trained PP with plots")
    common(t)
    t.add_argument("--model", nargs="+", choices=["edge", "gnn"])
    t.add_argument("--num", type=int, help="saved test instances (default: all)")
    t.add_argument("--random", type=int, metavar="K",
                   help="test on K new random scenarios instead of the saved ones")
    t.add_argument("--test_agents", type=int, help="agents in the random scenarios")
    t.add_argument("--seed", type=int, default=12345, help="seed of the random scenarios")
    t.add_argument("--no_cbs", action="store_true")
    t.add_argument("--animate", action="store_true")
    t.add_argument("--expert", choices=["eecbs", "cbs"])
    t.add_argument("--eecbs_bin")

    e = sub.add_parser("evaluate", help="Table 1 over all saved test scenarios, no plots")
    common(e)
    e.add_argument("--model", nargs="+", choices=["edge", "gnn"])
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
    models = getattr(args, "model", None) or ["edge", "gnn"]
    workers = args.workers or default_workers()

    if args.command == "generate":
        from mapf_nco.experts import generate_datasets

        generate_datasets(cfg, agents, workers=workers)

    if args.command == "experiment":
        from mapf_nco.train import train

        for n in agents:
            for kind in models:
                train(cfg, n, kind, workers=workers)

    if args.command == "test":
        from mapf_nco.evaluate import test_random, test_saved

        for n in agents:
            if args.random:
                test_random(cfg, n, args.test_agents or n, args.random, args.seed, models,
                            animate=args.animate, workers=workers)
            else:
                test_saved(cfg, n, args.num, models, animate=args.animate)

    if args.command == "evaluate":
        from mapf_nco.evaluate import evaluate, format_table, save_results

        rows = evaluate(cfg, agents, models)
        print("\n" + format_table(rows))
        csv_path, md_path = save_results(rows, os.path.join(cfg["output_dir"], "results"))
        print(f"\nSaved {csv_path} and {md_path}")


if __name__ == "__main__":
    sys.exit(main())
