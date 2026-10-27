"""Grid graph for MovingAI benchmark maps (paper Sec. 3.1).

The MAPF graph G = (V, E) has one vertex per traversable cell and edges between
4-neighbouring cells. Every vertex also gets a self-loop {v, v} that encodes a
wait action.

The paper's loss compares *directed* edge-usage vectors (Eq. 4) and perturbs the
cost vector with dL/dy (Eq. 5), so costs and usages must live in the same space.
We therefore index every directed move (u -> v) and every self-loop (v -> v)
as its own edge. ``num_edges`` is M in the paper.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# Fixed move order: up, right, down, left. Self-loop is appended last.
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
FREE_CHARS = {"."}  # EECBS treats only '.' as traversable; keep both solvers consistent.


class Grid:
    """A 4-connected grid graph with wait self-loops and directed edge indexing."""

    def __init__(self, rows, name="grid", path=None):
        self.name = name
        self.path = os.path.abspath(path) if path else None
        self.rows = list(rows)
        self.height = len(self.rows)
        self.width = len(self.rows[0]) if self.rows else 0
        if any(len(r) != self.width for r in self.rows):
            raise ValueError("All map rows must have the same width.")

        self.free = np.array(
            [[ch in FREE_CHARS for ch in row] for row in self.rows], dtype=bool
        )
        cells = [(r, c) for r in range(self.height) for c in range(self.width) if self.free[r, c]]
        self.coords = np.array(cells, dtype=np.int64).reshape(-1, 2)
        self.num_nodes = len(cells)
        self.node_of = {cell: i for i, cell in enumerate(cells)}

        src, dst = [], []
        self.neighbors = [[] for _ in range(self.num_nodes)]  # move neighbours (no self-loop)
        self.out_edges = [[] for _ in range(self.num_nodes)]  # (v, edge_id) incl. self-loop
        self.self_loop = np.zeros(self.num_nodes, dtype=np.int64)
        for u, (r, c) in enumerate(cells):
            for dr, dc in MOVES:
                v = self.node_of.get((r + dr, c + dc))
                if v is not None:
                    self.neighbors[u].append(v)
                    self.out_edges[u].append((v, len(src)))
                    src.append(u)
                    dst.append(v)
            self.self_loop[u] = len(src)
            self.out_edges[u].append((u, len(src)))
            src.append(u)
            dst.append(u)

        self.edge_src = np.array(src, dtype=np.int64)
        self.edge_dst = np.array(dst, dtype=np.int64)
        self.num_edges = len(src)
        self.is_self_loop = self.edge_src == self.edge_dst
        self.move_edges = np.flatnonzero(~self.is_self_loop)

        # Dense (u, v) -> edge id lookup; -1 marks an invalid move. N is ~800 for
        # random-32-32-20 so this is small and makes usage vectors vectorised.
        self.edge_matrix = np.full((self.num_nodes, self.num_nodes), -1, dtype=np.int64)
        self.edge_matrix[self.edge_src, self.edge_dst] = np.arange(self.num_edges)
        self.degree = np.array([len(n) for n in self.neighbors], dtype=np.int64)
        self._bfs_cache = {}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_file(cls, path):
        """Load a MovingAI ``.map`` file (``type/height/width/map`` header)."""
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\r\n") for ln in f]
        header = {}
        map_idx = None
        for i, ln in enumerate(lines):
            if ln.strip() == "map":
                map_idx = i
                break
            parts = ln.split()
            if len(parts) == 2:
                header[parts[0]] = parts[1]
        if map_idx is None:
            raise ValueError(f"{path}: missing 'map' line")
        height, width = int(header["height"]), int(header["width"])
        rows = lines[map_idx + 1 : map_idx + 1 + height]
        if len(rows) != height or any(len(r) < width for r in rows):
            raise ValueError(f"{path}: map dimensions do not match header")
        rows = [r[:width] for r in rows]
        name = os.path.splitext(os.path.basename(path))[0]
        return cls(rows, name=name, path=path)

    # ---------------------------------------------------------------- helpers
    def node(self, r, c):
        return self.node_of[(int(r), int(c))]

    def cell(self, v):
        r, c = self.coords[v]
        return int(r), int(c)

    def largest_component(self):
        """Node ids of the largest connected component."""
        seen = np.zeros(self.num_nodes, dtype=bool)
        best = []
        for s in range(self.num_nodes):
            if seen[s]:
                continue
            comp, q = [s], deque([s])
            seen[s] = True
            while q:
                u = q.popleft()
                for v in self.neighbors[u]:
                    if not seen[v]:
                        seen[v] = True
                        comp.append(v)
                        q.append(v)
            if len(comp) > len(best):
                best = comp
        return sorted(best)

    def bfs_distances(self, source):
        """Unit-cost shortest-path distances from ``source`` (np.inf if unreachable).

        The grid is undirected, so this is also the distance *to* ``source``.
        """
        cached = self._bfs_cache.get(source)
        if cached is not None:
            return cached
        dist = np.full(self.num_nodes, np.inf)
        dist[source] = 0
        q = deque([source])
        while q:
            u = q.popleft()
            for v in self.neighbors[u]:
                if dist[v] == np.inf:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if len(self._bfs_cache) < 4096:
            self._bfs_cache[source] = dist
        return dist

    def weighted_distances_to(self, targets, weights):
        """Shortest-path cost from every vertex to each target under ``weights``.

        Only move edges are used (waiting never shortens a path). Returns an
        array of shape (len(targets), num_nodes). This is the "true distance"
        heuristic of Sec. 3.3, recomputed under the current learned weights so
        that it stays admissible.
        """
        w = np.asarray(weights, dtype=np.float64)[self.move_edges]
        # csgraph treats explicit zeros as missing edges; callers clamp w >= w_min > 0.
        w = np.maximum(w, 1e-12)
        # Reverse graph: an edge u->v of cost w becomes v->u, so Dijkstra from the
        # target gives dist(x -> target) for every x.
        rev = csr_matrix(
            (w, (self.edge_dst[self.move_edges], self.edge_src[self.move_edges])),
            shape=(self.num_nodes, self.num_nodes),
        )
        return dijkstra(rev, directed=True, indices=np.asarray(targets, dtype=np.int64))
