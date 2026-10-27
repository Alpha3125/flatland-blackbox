"""MAPF NCO: replication of "Learned Representations Enhance Multi Agent Path
Planning" (Captari & van Hoof, ICML 2025 PRAL workshop).

Run without arguments for the interactive menu:

    python main.py                              (uses configs/paper.json)
    python main.py --config configs/smoke.json  (quick test settings)

Or use the commands directly:

    python main.py generate --agents 50              scenarios + EECBS expert plans
    python main.py train    --agents 50 --model gnn  train from scratch (loss graph, TensorBoard)
    python main.py test     --agents 50 --num 10     PP vs EECBS vs learned PP, comparison plots
    python main.py solve    --agents 50 --seed 1     one fresh random scenario, compared
    python main.py all      --agents 50              generate -> train -> test

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
            print(f"  {split:5s} data : {len(scs)} scenarios (expert: {meta.get('expert')})  {p}")
        else:
            info[split] = 0
            print(f"  {split:5s} data : not generated yet")
    for kind in ("edge", "gnn"):
        ck = checkpoint_path(cfg, grid.name, n, kind)
        info[kind] = os.path.exists(ck)
        print(f"  {kind:5s} model: {'trained  ' + ck if info[kind] else 'not trained yet'}")
    print(f"  outputs    : {run_dir(cfg, grid.name, n)}")
    return info


def menu_generate(cfg, n, info):
    from mapf_nco.experts import generate_datasets
    from mapf_nco.solvers.eecbs import find_eecbs

    cfg = dict(cfg)
    cfg["num_train"] = ask("Number of training scenarios (20% go to validation)",
                           cfg["num_train"], int)
    cfg["num_test"] = ask("Number of test scenarios", cfg["num_test"], int)
    if cfg.get("expert", "eecbs") == "eecbs" and find_eecbs(cfg.get("eecbs_bin")) is None:
        print("  EECBS was not found (see README: 'EECBS on Windows').")
        if not ask_yes("  Use the slow Python CBS instead (only OK for ~20 agents or fewer)?", False):
            return
        cfg["expert"] = "cbs"
    if info["train"] or info["test"]:
        cfg["overwrite"] = ask_yes("Data for this agent count exists. Replace it?", False)
    generate_datasets(cfg, [n], workers=default_workers())


def menu_train(cfg, n, info):
    from mapf_nco.paths import training_dir
    from mapf_nco.train import train

    if not info["train"]:
        print("  No training data yet: generate first (option 1).")
        return
    choice = ask("Model to train: 1) Edge  2) GNN  3) both", 3, int, [1, 2, 3])
    kinds = {1: ["edge"], 2: ["gnn"], 3: ["edge", "gnn"]}[choice]
    cfg = dict(cfg)
    cfg["epochs"] = ask("Epochs", cfg["epochs"], int)
    if ask_yes("Change batch size / learning rate / lambda?", False):
        cfg["batch_size"] = ask("Batch size", cfg["batch_size"], int)
        cfg["lr"] = ask("Learning rate", cfg["lr"], float)
        cfg["lam"] = ask("Lambda (perturbation strength)", cfg["lam"], float)
    for kind in kinds:
        print()
        train(cfg, n, kind, workers=default_workers())
        open_file(os.path.join(training_dir(cfg, info["grid"].name, n, kind), "curves.png"))
    print()
    if ask_yes("Test now?", True):
        menu_test(cfg, n, info)


def menu_test(cfg, n, info):
    from mapf_nco.evaluate import run_test
    from mapf_nco.paths import test_dir

    if not info["test"]:
        print("  No test data yet: generate first (option 1).")
        return
    k = ask(f"How many test instances (1-{info['test']})", info["test"], int,
            list(range(1, info["test"] + 1)))
    animate = ask_yes("Also save animations (GIF, slower)?", False)
    run_test(cfg, n, k, ("edge", "gnn"), plots=True, animate=animate)
    open_file(os.path.join(test_dir(cfg, info["grid"].name, n), "summary.png"))


def menu_fresh(cfg, n, info):
    from mapf_nco.compare import compare

    seed = ask("Random seed for the new scenario", 0, int)
    compare(cfg, n, seed=seed, kinds=("edge", "gnn"), animate=ask_yes("Animation?", False))


def interactive(config_path):
    cfg = load_config(config_path)
    print("=" * 64)
    print(" MAPF NCO: learned edge costs for Prioritized Planning")
    print(f" config: {config_path}")
    print("=" * 64)
    n = ask("Number of agents", cfg["agent_counts"][0], int)
    while True:
        info = status(cfg, n)
        print("""
  1) Generate scenarios + expert (EECBS) plans
  2) Train a model from scratch (Edge / GNN), then optionally test
  3) Test: PP vs EECBS vs trained PP on N test instances (comparison plots)
  4) Compare on one new random scenario
  5) Change number of agents
  0) Quit""")
        c = ask("Choose", None, int, [0, 1, 2, 3, 4, 5])
        try:
            if c == 0:
                return
            if c == 1:
                menu_generate(cfg, n, info)
            elif c == 2:
                menu_train(cfg, n, info)
            elif c == 3:
                menu_test(cfg, n, info)
            elif c == 4:
                menu_fresh(cfg, n, info)
            elif c == 5:
                n = ask("Number of agents", n, int)
        except KeyboardInterrupt:
            print("\n  (interrupted, back to menu)")
        except Exception as e:  # keep the menu alive, show what went wrong
            import traceback

            traceback.print_exc()
            print(f"\n  Error: {e}")


# --------------------------------------------------------------- commands
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", dest="menu_config", default=DEFAULT_CONFIG,
                   help="config for the interactive menu")
    sub = p.add_subparsers(dest="command")

    def common(sp):
        sp.add_argument("--config", default=DEFAULT_CONFIG)
        sp.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override config entries")
        sp.add_argument("--agents", type=int, nargs="+", help="agent counts (default: config)")
        sp.add_argument("--workers", type=int, default=None,
                        help="parallel processes (default: CPU count - 1)")

    g = sub.add_parser("generate", help="create scenarios and expert plans")
    common(g)
    g.add_argument("--expert", choices=["eecbs", "cbs"])
    g.add_argument("--eecbs_bin")
    g.add_argument("--overwrite", action="store_true")

    for name, hlp in (("train", "train models from scratch"),
                      ("all", "generate, train and test")):
        t = sub.add_parser(name, help=hlp)
        common(t)
        t.add_argument("--model", nargs="+", choices=["edge", "gnn"])
        t.add_argument("--epochs", type=int)
        t.add_argument("--batch_size", type=int)
        t.add_argument("--lr", type=float)
        t.add_argument("--lam", type=float)
        t.add_argument("--device")
        if name == "all":
            t.add_argument("--expert", choices=["eecbs", "cbs"])
            t.add_argument("--eecbs_bin")
            t.add_argument("--num", type=int, help="test instances to plot (default: all)")

    for name, hlp in (("test", "compare PP / EECBS / trained PP with plots (replaces test/)"),
                      ("evaluate", "Table 1 over all test instances for several agent counts")):
        e = sub.add_parser(name, help=hlp)
        common(e)
        e.add_argument("--model", nargs="+", choices=["edge", "gnn"])
        e.add_argument("--device")
        if name == "test":
            e.add_argument("--num", type=int, help="number of test instances (default: all)")
            e.add_argument("--animate", action="store_true")

    s = sub.add_parser("solve", help="compare methods on a single scenario")
    common(s)
    s.add_argument("--split", default="test", choices=["train", "test"])
    s.add_argument("--index", type=int, default=None)
    s.add_argument("--seed", type=int, default=None,
                   help="fresh random scenario with this seed instead of the dataset")
    s.add_argument("--animate", action="store_true")
    s.add_argument("--model", nargs="+", choices=["edge", "gnn"])
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

    explicit = {k: getattr(args, k, None) for k in
                ("expert", "eecbs_bin", "epochs", "batch_size", "lr", "lam", "device")}
    if getattr(args, "overwrite", False):
        explicit["overwrite"] = True
    cfg = load_config(args.config, args.set, **explicit)
    agents = args.agents or cfg["agent_counts"]
    models = getattr(args, "model", None) or cfg.get("models", ["edge", "gnn"])
    workers = args.workers or default_workers()

    if args.command in ("generate", "all"):
        from mapf_nco.experts import generate_datasets

        generate_datasets(cfg, agents, workers=workers)

    if args.command in ("train", "all"):
        from mapf_nco.train import train

        for n in agents:
            for kind in models:
                train(cfg, n, kind, workers=workers)

    if args.command in ("test", "all"):
        from mapf_nco.evaluate import run_test

        for n in agents:
            run_test(cfg, n, getattr(args, "num", None), models, plots=True,
                     animate=getattr(args, "animate", False))

    if args.command == "evaluate":
        from mapf_nco.evaluate import evaluate, format_table, save_results

        rows = evaluate(cfg, agents, models)
        print("\n" + format_table(rows))
        csv_path, md_path = save_results(rows, os.path.join(cfg["output_dir"], "results"))
        print(f"\nSaved {csv_path} and {md_path}")

    if args.command == "solve":
        from mapf_nco.compare import compare

        compare(cfg, agents[0], args.split, args.index, args.seed, models, args.animate)


if __name__ == "__main__":
    sys.exit(main())
