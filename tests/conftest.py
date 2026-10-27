import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MAP_PATH = os.path.join(ROOT, "data", "random-32-32-20.map")


@pytest.fixture(scope="session")
def grid():
    from mapf_nco.grid import Grid

    return Grid.from_file(MAP_PATH)


@pytest.fixture
def open_grid():
    """A 5x5 obstacle-free grid with many equal-length paths."""
    from mapf_nco.grid import Grid

    return Grid(["....."] * 5, name="open5")
