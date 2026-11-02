"""Where everything is written.

    data/scenarios/<map>/n<N>/{train,test}.json.gz     generated inputs (scenarios + EECBS plans)
    outputs/<map>/n<N>/
        checkpoints/{edge,gnn}.pt                      trained PP models (best validation epoch)
        checkpoints/{edge,gnn}_cbs.pt                  trained CBS models
        training/                                      curves.png (all models together),
                                                       history.csv, history.json
        tensorboard/<model>/, tensorboard/test/        TensorBoard logs
        test/                                          comparison plots + tables (replaced on every test)
        test_n<M>/                                     second test set of a sweep
    outputs/sweeps/exp_<k>/                           sweeps, k = smallest free number
                                                       (one exp_XX_<settings>/ each)
"""

from __future__ import annotations

import os
import shutil


def run_dir(cfg, map_name, n):
    return os.path.join(cfg["output_dir"], map_name, f"n{n}")


def checkpoint_path(cfg, map_name, n, kind):
    return os.path.join(run_dir(cfg, map_name, n), "checkpoints", f"{kind}.pt")


def training_dir(cfg, map_name, n):
    return os.path.join(run_dir(cfg, map_name, n), "training")


def tensorboard_dir(cfg, map_name, n, name):
    return os.path.join(run_dir(cfg, map_name, n), "tensorboard", name)


def test_dir(cfg, map_name, n, name="test"):
    return os.path.join(run_dir(cfg, map_name, n), name)


def fresh_dir(path):
    """Delete ``path`` (if it exists) and recreate it empty."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path
