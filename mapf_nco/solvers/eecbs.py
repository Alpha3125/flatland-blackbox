"""Wrapper around the official EECBS binary (Li, Ruml & Koenig, AAAI 2021).

The paper generates expert plans with EECBS (suboptimality 1.0 for 50 agents and
1.05 for 75/100 agents). Build it from https://github.com/Jiaoyang-Li/EECBS
(see ``scripts/build_eecbs.sh``) and point ``--eecbs_bin`` or the
``MAPF_NCO_EECBS`` environment variable at the executable.

Windows: build EECBS inside WSL (``bash scripts/build_eecbs.sh`` from the repo
folder). The resulting Linux binary is detected automatically and run through
``wsl -e``, with file paths translated (C:\\... -> /mnt/c/...), so the rest of
the pipeline keeps running in your normal Windows Python. A binary that lives
only inside the WSL file system can be given as ``--eecbs_bin wsl:/home/me/eecbs``.
A native ``eecbs.exe`` (built with Visual Studio + vcpkg Boost) also works.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

from mapf_nco.scenarios import write_scen

IS_WINDOWS = sys.platform == "win32"

_DEFAULT_LOCATIONS = [
    os.path.join("third_party", "EECBS", "build", "Release", "eecbs.exe"),
    os.path.join("third_party", "EECBS", "build", "eecbs.exe"),
    os.path.join("third_party", "EECBS", "build", "eecbs"),
    os.path.join("third_party", "EECBS", "eecbs"),
]

_CELL_RE = re.compile(r"\((\d+),(\d+)\)")


def _is_linux_elf(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def uses_wsl(binary):
    """True if ``binary`` must be run through WSL from Windows."""
    return binary.startswith("wsl:") or (IS_WINDOWS and _is_linux_elf(binary))


def to_wsl_path(path):
    """Translate a Windows path to the path WSL sees (C:\\x\\y -> /mnt/c/x/y)."""
    p = os.path.abspath(path) if IS_WINDOWS else path
    p = p.replace("\\", "/")
    for prefix in ("//wsl.localhost/", "//wsl$/"):  # \\wsl$\\Ubuntu\\home\\me -> /home/me
        if p.lower().startswith(prefix.lower()):
            rest = p[len(prefix):]
            return "/" + rest.split("/", 1)[1] if "/" in rest else "/"
    if len(p) >= 2 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def find_eecbs(explicit=None):
    """Locate the EECBS executable, or return None."""
    candidates = [explicit, os.environ.get("MAPF_NCO_EECBS")]
    candidates += [shutil.which("eecbs")]
    candidates += _DEFAULT_LOCATIONS
    for c in candidates:
        if not c:
            continue
        if c.startswith("wsl:"):
            return c
        if not os.path.isfile(c):
            continue
        if (IS_WINDOWS and _is_linux_elf(c)) or os.access(c, os.X_OK):
            return os.path.abspath(c)
    return None


def _command(binary):
    """Command prefix and path converter for running ``binary``."""
    if uses_wsl(binary):
        if shutil.which("wsl") is None:
            raise FileNotFoundError(
                f"{binary} is a Linux program but 'wsl' is not available. Install WSL "
                "(wsl --install) or build a native eecbs.exe.")
        target = binary[4:] if binary.startswith("wsl:") else to_wsl_path(binary)
        return ["wsl", "-e", target], to_wsl_path
    return [binary], (lambda p: p)


def check_eecbs(grid, binary, num_agents=10, time_limit=20.0):
    """Run EECBS once on an easy random scenario. Returns (ok, message)."""
    import random

    cells = random.Random(0).sample(grid.largest_component(), 2 * num_agents)
    paths, err = solve_eecbs(grid, cells[:num_agents], cells[num_agents:], binary,
                             1.0, time_limit, return_error=True)
    if paths is None:
        return False, f"EECBS at {binary} did not return a solution.\n{err}".strip()
    return True, f"EECBS OK ({binary})"


def parse_eecbs_paths(text, grid):
    """Parse the ``--outputPaths`` file: ``Agent i: (r,c)->(r,c)->...``."""
    paths = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("Agent"):
            continue
        head, _, body = line.partition(":")
        i = int(head.split()[1])
        paths[i] = [grid.node(int(r), int(c)) for r, c in _CELL_RE.findall(body)]
    return [paths[i] for i in range(len(paths))]


def solve_eecbs(
    grid,
    starts,
    goals,
    binary,
    suboptimality=1.0,
    time_limit=60.0,
    extra_args=None,
    return_error=False,
):
    """Run EECBS on one scenario. Returns a list of paths or None on failure.

    With ``return_error=True`` returns ``(paths, error_text)`` instead.
    """
    if grid.path is None:
        raise ValueError("EECBS needs the grid to be loaded from a .map file")
    with tempfile.TemporaryDirectory(prefix="mapf_nco_eecbs_") as tmp:
        scen = os.path.join(tmp, "agents.scen")
        out_paths = os.path.join(tmp, "paths.txt")
        stats = os.path.join(tmp, "stats.csv")
        write_scen(scen, grid, starts, goals)
        prefix, conv = _command(binary)
        cmd = prefix + [
            "-m", conv(grid.path),
            "-a", conv(scen),
            "-k", str(len(starts)),
            "-t", str(time_limit),
            "-o", conv(stats),
            "--outputPaths", conv(out_paths),
            "--suboptimality", str(suboptimality),
            "-s", "0",
        ]
        cmd += list(extra_args or [])
        err = ""
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=time_limit + 60,
                check=False,
            )
            err = (proc.stderr or b"").decode(errors="replace")
            if not os.path.exists(out_paths):
                out = (proc.stdout or b"").decode(errors="replace")
                err = f"exit code {proc.returncode}\ncommand: {' '.join(cmd)}\n{out[-800:]}\n{err[-800:]}"
        except subprocess.TimeoutExpired:
            err = "timed out"
        paths = None
        if os.path.exists(out_paths):  # EECBS only writes paths when it found a solution
            with open(out_paths, "r", encoding="utf-8") as f:
                paths = parse_eecbs_paths(f.read(), grid)
            if len(paths) != len(starts):
                paths = None
    return (paths, err) if return_error else paths
