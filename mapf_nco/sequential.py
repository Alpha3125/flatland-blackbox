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
