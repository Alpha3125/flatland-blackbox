"""Sequential (Gelato-style) training: learn at every state where PP leaves the expert's plan.

The paper's training compares PP's whole plan of a scenario with the expert's. The
sequential method walks each training scenario along the expert's (EECBS/CBS) plan,
one timestep (state) at a time:

    for t = 0, 1, ..., (expert makespan - 1):
        state s_t   = every agent's position at time t in the expert's plan
        PP plans all agents from s_t to their goals (same priority order, current costs)
        if PP's next move differs from the expert's for any agent (a deviation):
            loss  = the usual loss between PP's remaining plan from s_t and the expert's
                    remaining plan (usage: the paper's Eq. 4 on the remaining edge usage)
            grad  = the paper's black-box gradient for it (Eqs. 5-6)
            -> update the costs (``seq_update``: "state" = an optimizer step right away,
               "batch" = summed over the batch's deviating states, one step per batch)
        continue from s_{t+1} of the expert's plan (teacher forcing), whatever PP did

So the costs are corrected exactly where PP goes wrong, and the states where PP
already agrees with the expert give no update. Validation and testing are unchanged:
plain PP plans the whole scenario once with the learned costs.

``seq_mode: blockade`` (:class:`Blockade`) replaces the gradient step by an
aggressive one: PP's deviating moves get much dearer (up to a near-blockade), the
expert's cheaper, and PP replans from the same state until it follows the expert.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mapf_nco.blackbox import blackbox_step
from mapf_nco.metrics import edge_usage
from mapf_nco.solver_pp import NoSolutionError

SEQ_UPDATES = ("state", "batch")


def expert_states(grid, sc):
    """[(t, positions at t, positions at t+1, expert usage from t on)] along the expert plan."""
    paths = sc.expert_paths
    T = max(len(p) for p in paths) - 1

    def at(p, t):
        return p[min(t, len(p) - 1)]

    out = []
    for t in range(T):
        rest = [p[t:] if t < len(p) else [p[-1]] for p in paths]
        out.append((t, [at(p, t) for p in paths], [at(p, t + 1) for p in paths],
                    edge_usage(grid, rest)))
    return out


def next_moves(paths):
    return [p[1] if len(p) > 1 else p[0] for p in paths]


def state_step(planner, state, goals, order, w, lam, reduction, obj):
    """One state: (deviated, BlackBoxResult or None). Plans PP once from the state; only a
    deviation (some agent's next move differs from the expert's) gets the gradient."""
    _, pos, nxt, y_rest = state
    try:
        paths = planner.solve(pos, goals, order, w)
    except NoSolutionError:
        return None, None  # PP can't plan from this state with this order
    if next_moves(paths) == list(nxt):
        return False, None
    r = blackbox_step(planner, pos, goals, order, w, y_rest, lam, reduction, paths=paths, **obj)
    return True, r


def scenario_steps(planner, grid, sc, w, lam, reduction, obj):
    """All states of one scenario with frozen costs ``w`` (the "batch" update)."""
    grad = np.zeros(len(w))
    losses, n_dev, n_fail = [], 0, 0
    states = expert_states(grid, sc)
    for st in states:
        dev, r = state_step(planner, st, sc.goals, sc.order, w, lam, reduction, obj)
        if dev is None:
            n_fail += 1
        elif dev and r is not None and r.ok:
            n_dev += 1
            grad += r.grad
            losses.append(r.loss)
    return SeqResult(grad, float(np.mean(losses)) if losses else float("nan"), len(states),
                     n_dev, n_fail)


@dataclass
class SeqResult:
    grad: np.ndarray  # summed over the deviating states
    loss: float  # mean over the deviating states (nan if none)
    states: int
    deviations: int
    failures: int  # states PP could not plan from

    @property
    def ok(self):
        return self.deviations > 0


# ------------------------------------------------------------------- blockade mode
SEQ_MODES = ("gradient", "blockade")
BLOCKADE_KINDS = ("edge", "potential_edge", "cell")  # a cost per edge or per cell


class Blockade:
    """The aggressive sequential update (``seq_mode: blockade``).

    At a state where PP's next moves deviate from the expert's, the moves PP makes
    instead (for the cell model: the cells it enters instead) get ``factor`` times
    dearer, up to ``cap`` (a near-blockade: still passable, so PP never gets stuck),
    and the expert's moves get ``factor`` times cheaper, down to 1. PP replans from
    the same state until it follows the expert or ``retries`` updates are used up.
    After every epoch, ``relax`` pulls every cost back towards 1 (cost - 1 shrinks
    by that factor), so only blockades that keep being needed survive.

    The costs live in ``p``: per edge for the Edge model (cost = p), per cell for the
    cell model (cost of an edge = 1 + p[destination cell]).
    """

    def __init__(self, grid, kind, factor=3.0, cap=20.0, retries=5):
        if kind not in BLOCKADE_KINDS:
            raise ValueError(f"the blockade mode needs a cost per edge or per cell "
                             f"({', '.join(BLOCKADE_KINDS)}), not {kind!r}")
        if factor <= 1 or cap <= 1:
            raise ValueError("block_factor and block_cap must be > 1")
        self.grid, self.cell = grid, kind == "cell"
        self.factor, self.cap, self.retries = float(factor), float(cap), int(retries)

    def costs(self, p):
        return 1.0 + p[self.grid.edge_dst] if self.cell else p

    def _index(self, edges):
        return np.unique(self.grid.edge_dst[edges] if self.cell else np.asarray(edges))

    def _get(self, p, idx):
        return 1.0 + p[idx] if self.cell else p[idx]

    def _set(self, p, idx, c):
        p[idx] = c - 1.0 if self.cell else c

    def update(self, p, pp_edges, expert_edges):
        """Dearer PP's moves (not those the expert also makes), cheaper the expert's."""
        good = self._index(expert_edges)
        bad = np.setdiff1d(self._index(pp_edges), good)
        self._set(p, good, np.maximum(1.0, self._get(p, good) / self.factor))
        self._set(p, bad, np.minimum(self.cap, np.maximum(1.0, self._get(p, bad)) * self.factor))
        return len(bad)

    def relax(self, p, keep):
        """Cost - 1 times ``keep`` (0.9 = 10% of every blockade fades per epoch)."""
        if self.cell:
            p *= keep
        else:
            p[:] = 1.0 + (p - 1.0) * keep

    def state(self, planner, st, goals, order, p):
        """One state: {deviated, fixed, failed, updates}; changes ``p`` in place."""
        _, pos, nxt, _ = st
        em = self.grid.edge_matrix
        out = {"deviated": False, "fixed": False, "failed": False, "updates": 0}
        for attempt in range(self.retries + 1):
            try:
                paths = planner.solve(pos, goals, order, self.costs(p))
            except NoSolutionError:
                out["failed"] = True
                return out
            ppn = next_moves(paths)
            bad = [i for i in range(len(pos)) if ppn[i] != nxt[i]]
            if not bad:
                out["fixed"] = out["deviated"]
                return out
            out["deviated"] = True
            if attempt == self.retries:
                return out
            out["updates"] += 1
            self.update(p, [em[pos[i], ppn[i]] for i in bad], [em[pos[i], nxt[i]] for i in bad])
        return out
