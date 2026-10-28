"""Where everything is written.

    data/scenarios/<map>/n<N>/{train,test}.json.gz     generated inputs (scenarios + EECBS plans)
    outputs/<map>/n<N>/
        checkpoints/{edge,gnn}.pt                      trained models (best validation epoch)
        training/{edge,gnn}/                           curves.png, history.csv, history.json
        tensorboard/{edge,gnn,test}/                   TensorBoard logs
        test/                                          comparison plots + tables (replaced on every test)
        plots/                                         single-scenario comparisons (main.py solve)
"""

from __future__ import annotations

import os
import shutil


def run_dir(cfg, map_name, n):
    return os.path.join(cfg["output_dir"], map_name, f"n{n}")


def checkpoint_path(cfg, map_name, n, kind):
    return os.path.join(run_dir(cfg, map_name, n), "checkpoints", f"{kind}.pt")


def training_dir(cfg, map_name, n, kind):
    return os.path.join(run_dir(cfg, map_name, n), "training", kind)


def tensorboard_dir(cfg, map_name, n, name):
    return os.path.join(run_dir(cfg, map_name, n), "tensorboard", name)


def test_dir(cfg, map_name, n, name="test"):
    return os.path.join(run_dir(cfg, map_name, n), name)


def plots_dir(cfg, map_name, n):
    return os.path.join(run_dir(cfg, map_name, n), "plots")


def fresh_dir(path):
    """Delete ``path`` (if it exists) and recreate it empty."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path
