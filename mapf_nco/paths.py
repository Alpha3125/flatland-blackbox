"""Where everything is written. Every file starts with the name of its folder.

    data/scenarios/<map>/n<N>/{train,test}.json.gz     generated inputs (scenarios + EECBS plans)
    outputs/<map>/n<N>/
        models/model_edge.pt                           trained PP Edge model (best validation epoch)
        models/model_edge_cbs.pt                       trained CBS Edge model
        training/                                      training_curves.png (all models together),
                                                       training_history.csv, training_history.json
        tensorboard/tensorboard_<model>/,              TensorBoard logs (the event files inside are
        tensorboard/tensorboard_test/                  named by TensorBoard)
        test/                                          test_summary.*, test_per_instance.csv,
                                                       test_instance_000_comparison.png, ...
                                                       (replaced on every test)
        test_n<M>/                                     second test set of a sweep (same names)
    outputs/sweeps/exp_<k>/                           sweeps, k = smallest free number
                                                       (one exp_XX_<settings>/ each)

Models trained before this layout (checkpoints/<model>.pt) are still found and loaded.
"""

from __future__ import annotations

import os
import shutil


def run_dir(cfg, map_name, n):
    return os.path.join(cfg["output_dir"], map_name, f"n{n}")


def checkpoint_path(cfg, map_name, n, kind):
    """Where model ``kind`` (a model id, e.g. edge_cbs) is saved."""
    return os.path.join(run_dir(cfg, map_name, n), "models", f"model_{kind}.pt")


def legacy_checkpoint_path(cfg, map_name, n, kind):
    """Where models were saved before the current layout."""
    return os.path.join(run_dir(cfg, map_name, n), "checkpoints", f"{kind}.pt")


def find_checkpoint(cfg, map_name, n, kind):
    """The saved model ``kind`` (current layout first, then the old one), or None."""
    for path in (checkpoint_path(cfg, map_name, n, kind),
                 legacy_checkpoint_path(cfg, map_name, n, kind)):
        if os.path.exists(path):
            return path
    return None


def training_dir(cfg, map_name, n):
    return os.path.join(run_dir(cfg, map_name, n), "training")


def training_file(cfg, map_name, n, name):
    """training/training_<name>, e.g. training_file(..., "curves.png")."""
    return os.path.join(training_dir(cfg, map_name, n), f"training_{name}")


def tensorboard_dir(cfg, map_name, n, name):
    """TensorBoard logs of a model (name = its id) or a test (name = test, test_n<M>)."""
    return os.path.join(run_dir(cfg, map_name, n), "tensorboard", f"tensorboard_{name}")


def test_dir(cfg, map_name, n, name="test"):
    return os.path.join(run_dir(cfg, map_name, n), name)


def test_file(out_dir, name):
    """A file of a test folder: test_<name>, e.g. test_instance_000_comparison.png."""
    return os.path.join(out_dir, f"test_{name}")


def fresh_dir(path):
    """Delete ``path`` (if it exists) and recreate it empty."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path
