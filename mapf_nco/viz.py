"""Pictures and animations of plans (needs matplotlib)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FADED = (0.6, 0.6, 0.6, 0.35)


def _agent_color(i):
    return plt.get_cmap("tab20")(i % 20)


def _draw_map(ax, grid):
    ax.imshow(grid.free, cmap="gray", origin="upper", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])


def _xy(grid, path):
    rc = [grid.cell(v) for v in path]
    return [c for _, c in rc], [r for r, _ in rc]


def _save(fig, out_path, dpi=150):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_paths(grid, paths, starts, goals, title, out_path):
    """All agents' paths on one map, one colour per agent (o = start, * = goal)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_map(ax, grid)
    for i, p in enumerate(paths):
        color = _agent_color(i)
        xs, ys = _xy(grid, p)
        ax.plot(xs, ys, color=color, lw=1.5, alpha=0.85)
        ax.scatter(xs[:1], ys[:1], color=color, marker="o", s=25, edgecolors="black", zorder=3)
        ax.scatter(xs[-1:], ys[-1:], color=color, marker="*", s=70, edgecolors="black", zorder=3)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out_path)


def changed_agents(plans, reference):
    """Agents whose path differs from ``plans[reference]`` in each other plan."""
    ref = plans[reference]
    out = {}
    for name, paths in plans.items():
        if name != reference and paths is not None:
            out[name] = {i for i, (a, b) in enumerate(zip(ref, paths)) if list(a) != list(b)}
    out[reference] = set().union(*out.values()) if out else set()
    return out


def _draw_plans(ax, grid, paths, changed):
    for i, p in enumerate(paths):
        xs, ys = _xy(grid, p)
        if i in changed:
            c = _agent_color(i)
            ax.plot(xs, ys, color=c, lw=2.2, zorder=2)
            ax.scatter(xs[:1], ys[:1], color=c, marker="o", s=30, edgecolors="black", zorder=4)
            ax.scatter(xs[-1:], ys[-1:], color=c, marker="*", s=90, edgecolors="black", zorder=4)
            ax.annotate(str(i), (xs[0], ys[0]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points", zorder=5)
        else:
            ax.plot(xs, ys, color=FADED, lw=1.0, zorder=1)


def weight_norm(weight_list):
    """Shared colour scale for edge costs, centred on the original cost 1."""
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    allw = np.concatenate([np.asarray(w, float) for w in weight_list])
    return TwoSlopeNorm(vcenter=1.0, vmin=min(float(allw.min()), 0.999),
                        vmax=max(float(allw.max()), 1.001))


def draw_weights(ax, grid, w, norm, cmap="coolwarm"):
    """Edges as segments coloured by cost (mean of both directions); waits as squares."""
    import numpy as np
    from matplotlib.collections import LineCollection

    w = np.asarray(w, dtype=float)
    segs, vals = [], []
    for e in grid.move_edges:
        u, v = int(grid.edge_src[e]), int(grid.edge_dst[e])
        if u < v:
            (r1, c1), (r2, c2) = grid.cell(u), grid.cell(v)
            segs.append([(c1, r1), (c2, r2)])
            vals.append(0.5 * (w[e] + w[grid.edge_matrix[v, u]]))
    _draw_map(ax, grid)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=3.0, zorder=2)
    lc.set_array(np.array(vals))
    ax.add_collection(lc)
    ax.scatter(grid.coords[:, 1], grid.coords[:, 0], c=w[grid.self_loop], cmap=cmap, norm=norm,
               marker="s", s=9, linewidths=0, zorder=3)
    return np.array(vals)


def plot_comparison(grid, sc, plans, titles, out_path, reference, weights=None, header=None,
                    weight_titles=None):
    """Comparison figure.

    Top row: one map per method. Agents whose path differs from the reference
    (original PP) are coloured and numbered; unchanged agents are grey.
    Bottom row (if ``weights``): the edge costs each model gives this scenario,
    as a blue (cheaper than 1) to red (dearer) gradient, with the starts (o) and
    goals (*). ``header`` (e.g. the training hyperparameters) is printed on top.
    """
    names = [k for k, v in plans.items() if v is not None]
    changed = changed_agents({k: plans[k] for k in names}, reference)
    weights = weights or {}
    rows = 2 if weights else 1
    cols = max(len(names), len(weights))
    fig, axes = plt.subplots(rows, cols, figsize=(4.9 * cols, 5.3 * rows + 0.6), squeeze=False)
    for ax, name in zip(axes[0], names):
        _draw_map(ax, grid)
        _draw_plans(ax, grid, plans[name], changed[name])
        ax.set_title(titles[name], fontsize=10)
    for ax in axes[0][len(names):]:
        ax.axis("off")
    if weights:
        norm = weight_norm(list(weights.values()))
        for ax, (kind, w) in zip(axes[1], weights.items()):
            vals = draw_weights(ax, grid, w, norm)
            for i, (s, g) in enumerate(zip(sc.starts, sc.goals)):
                (sr, scol), (gr, gc) = grid.cell(s), grid.cell(g)
                ax.scatter([scol], [sr], color=_agent_color(i), marker="o", s=22,
                           edgecolors="black", zorder=5)
                ax.scatter([gc], [gr], color=_agent_color(i), marker="*", s=60,
                           edgecolors="black", zorder=5)
            label = (weight_titles or {}).get(kind, kind)
            ax.set_title(f"{label}: min {vals.min():.2f}, max {vals.max():.2f}", fontsize=10)
        for ax in axes[1][len(weights):]:
            ax.axis("off")
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="coolwarm"),
                          ax=list(axes[1][:len(weights)]), fraction=0.03, pad=0.02)
        cb.set_label("edge cost (blue = cheaper, red = dearer than original 1)")
    top = "Top: coloured = agents whose path differs from the original PP plan, grey = unchanged."
    if weights:
        top += "  Bottom: edge costs the models assign (gradient)."
    fig.suptitle((header + "\n" if header else "") + top, fontsize=9)
    if not weights:
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.subplots_adjust(top=0.9, hspace=0.12, wspace=0.05, left=0.02, right=0.97,
                            bottom=0.02)
    return _save(fig, out_path, dpi=120)


def animate_comparison(grid, sc, plans, titles, out_path, fps=4):
    """GIF of all methods side by side, agents moving one step per frame."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    names = [k for k, v in plans.items() if v is not None]
    T = max(len(p) for k in names for p in plans[k])
    n = len(sc.starts)
    colors = [_agent_color(i) for i in range(n)]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.6), squeeze=False)
    dots, labels = [], []
    for ax, name in zip(axes[0], names):
        _draw_map(ax, grid)
        for i, p in enumerate(plans[name]):
            xs, ys = _xy(grid, p)
            ax.plot(xs, ys, color=colors[i], lw=0.8, alpha=0.3)
            ax.scatter(xs[-1:], ys[-1:], color=colors[i], marker="*", s=40, alpha=0.6)
        sc_plot = ax.scatter([0] * n, [0] * n, c=colors, s=45, edgecolors="black", zorder=5)
        dots.append((name, sc_plot))
        labels.append(ax.set_title(titles[name], fontsize=9))
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def frame(t):
        for name, sc_plot in dots:
            pos = []
            for p in plans[name]:
                r, c = grid.cell(p[min(t, len(p) - 1)])
                pos.append((c, r))
            sc_plot.set_offsets(pos)
        fig.suptitle(f"t = {t}", fontsize=10)
        return [d for _, d in dots]

    anim = FuncAnimation(fig, frame, frames=T, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


LOSS_FORMULA = {
    "mean": r"$L = \frac{1}{M}\sum_{e} (\hat y_e - y^*_e)^2$",
    "sum": r"$L = \sum_{e} (\hat y_e - y^*_e)^2$",
}
MAKESPAN_FORMULA = r"$\tau \log \sum_i e^{T_i/\tau}$,  $T_i = \sum_e \hat y_{i,e}$"


def loss_title(refs, reduction):
    obj = refs.get("objective", "usage")
    if obj == "makespan":
        return "Training loss (smooth makespan)   " + MAKESPAN_FORMULA + f",  $\\tau$={refs['tau']}"
    if obj == "usage+makespan":
        return ("Training loss: usage + " + f"{refs['beta']}" + r" $\times$ makespan" + "\n"
                + LOSS_FORMULA.get(reduction, "") + "  +  " + MAKESPAN_FORMULA)
    return "Training loss   " + LOSS_FORMULA.get(reduction, "")


MODEL_COLORS = {"edge": ("#4292c6", "#c6dbef"), "gnn": ("#ec7014", "#fdd0a2"),
                "edge_cbs": ("#238b45", "#c7e9c0"), "gnn_cbs": ("#88419d", "#dadaeb")}
RUN_LABELS = {"edge": "PP (Edge)", "gnn": "PP (GNN)", "edge_cbs": "CBS (Edge)",
              "gnn_cbs": "CBS (GNN)"}


def plot_training_curves(runs, n, out_path):
    """All models' training together: loss (left) and the validation metric (right).

    ``runs``: {model id: {history, batch_losses, refs, best_epoch, reduction}}, model
    ids edge / gnn (PP) and edge_cbs / gnn_cbs (CBS).
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    titles, ref_drawn = [], set()
    for kind, r in runs.items():
        dark, light = MODEL_COLORS.get(kind, ("black", "gray"))
        name = RUN_LABELS.get(kind, kind)
        refs, history = r["refs"], r["history"]
        metric = refs.get("metric", "delay")
        epochs = [h["epoch"] for h in history]
        if r.get("batch_losses"):
            bx, by = zip(*r["batch_losses"])
            a1.plot(bx, by, color=light, lw=0.8)
        pts = [(h["epoch"], h.get("train_loss")) for h in history if h.get("train_loss") is not None]
        if pts:
            a1.plot(*zip(*pts), "o-", color=dark, lw=2, label=f"{name}: epoch mean (light: batches)")
        val = [h[f"val_{metric}"] for h in history]
        a2.plot(epochs, val, "o-", color=dark, lw=2, label=name)
        be = r["best_epoch"]
        a2.scatter([be], [val[epochs.index(be)]], s=170, marker="*", color="gold",
                   edgecolors=dark, linewidths=1.5, zorder=5, label=f"{name} selected (epoch {be})")
        pl = refs.get("planner", "pp")
        key = (metric, pl, round(refs["orig_pp"], 6))
        if key not in ref_drawn:
            ref_drawn.add(key)
            a2.axhline(refs["orig_pp"], color="gray" if pl == "pp" else "#6a51a3", ls="--",
                       label=f"original {pl.upper()} ({refs['orig_pp']:.2f})")
        ekey = (metric, "expert", round(refs["expert"], 6))
        if ekey not in ref_drawn:
            ref_drawn.add(ekey)
            a2.axhline(refs["expert"], color="green", ls=":",
                       label=f"{refs.get('expert_name', 'EECBS')} ({refs['expert']:.2f})")
        titles.append(f"{name}: method {refs.get('method', 'static')}")
    first = next(iter(runs.values()))
    metric = first["refs"].get("metric", "delay")
    a1.set_xlabel("epoch")
    a1.set_ylabel("loss")
    a1.set_title(loss_title(first["refs"], first.get("reduction", "mean")), fontsize=10)
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    a2.set_xlabel("epoch")
    a2.set_ylabel(r"validation delay  ($\sum_e y_e$ $-$ lower bound)" if metric == "delay"
                  else "validation makespan (max arrival time)")
    a2.set_title(f"Validation {metric} (lower is better)", fontsize=11)
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    obj = first["refs"].get("objective", "usage")
    fig.suptitle(f"{n} agents, objective {obj}  |  " + "  |  ".join(titles), fontsize=10)
    fig.tight_layout()
    return _save(fig, out_path, dpi=130)


METHOD_COLORS = {"PP (orig)": "#969696", "PP (Edge)": "#4292c6", "PP (GNN)": "#ec7014",
                 "CBS (Edge)": "#238b45", "CBS (GNN)": "#88419d",
                 "Expert": "#41ab5d", "CBS": "#6a51a3"}


def plot_test_summary(per_instance, methods, out_path, title, expert_name="EECBS"):
    """Grouped bars: delay of every method on every test instance, plus the mean."""
    import numpy as np

    k = len(per_instance)
    labels = [str(r["instance"]) for r in per_instance] + ["mean"]
    x = np.arange(k + 1)
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * (k + 1) * len(methods) + 2), 4.8))
    for j, m in enumerate(methods):
        vals = [r.get(m) for r in per_instance]
        known = [v for v in vals if v is not None]
        vals = [np.nan if v is None else v for v in vals] + [np.mean(known) if known else np.nan]
        ax.bar(x + (j - (len(methods) - 1) / 2) * width, vals, width,
               label=expert_name if m == "Expert" else m, color=METHOD_COLORS.get(m))
    ax.axvline(k - 0.5, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("test instance")
    ax.set_ylabel(r"delay  ($\sum_e y_e$ $-$ lower bound)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_path, dpi=130)


def plot_sweep_summary(rows, tags, out_path):
    """Test delay of every experiment on each test set, with PP and expert reference lines."""
    import numpy as np

    names = [r["experiment"][4:] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, len(tags), figsize=(max(7, 1.1 * len(rows) + 4) * len(tags) / 1.4,
                                                    4.6), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        ids = [m for m in RUN_LABELS
               if any(np.isfinite(r.get(f"{tag}_{m}_delay", float("nan"))) for r in rows)]
        width = 0.8 / max(1, len(ids))
        for j, m in enumerate(ids):
            ax.bar(x + (j - (len(ids) - 1) / 2) * width,
                   [r.get(f"{tag}_{m}_delay", float("nan")) for r in rows], width,
                   label=RUN_LABELS[m], color=METHOD_COLORS[RUN_LABELS[m]])
        ax.axhline(rows[0][f"{tag}_orig_delay"], color="gray", ls="--", label="original PP")
        cbs = rows[0].get(f"{tag}_cbs_delay", float("nan"))
        if np.isfinite(cbs):
            ax.axhline(cbs, color="#6a51a3", ls="-.", label="original CBS")
        ax.axhline(rows[0][f"{tag}_expert_delay"], color="green", ls=":", label="EECBS")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel("experiment (see sweep_summary.md for its settings)")
        ax.set_ylabel(r"delay  ($\sum_e y_e$ $-$ lower bound)")
        ax.set_title(f"test set, {tag[1:]} agents", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Sweep: models trained on {tags[0][1:]} agents (lower is better)", fontsize=11)
    fig.tight_layout()
    return _save(fig, out_path, dpi=130)


def _evolution_panels(grid, sc, W, paths, picks, axes, norm):
    import numpy as np

    order = list(sc.order)
    for ax, k in zip(axes, picks):
        i = order[k]
        draw_weights(ax, grid, W[i], norm)
        for j in order[:k]:  # agents already planned (their reservations)
            xs, ys = _xy(grid, paths[j])
            ax.plot(xs, ys, color="black", lw=0.7, alpha=0.6, zorder=3)
        if paths[i] is not None:
            xs, ys = _xy(grid, paths[i])
            ax.plot(xs, ys, color="lime", lw=2.2, zorder=4)
        (sr, scol), (gr, gc) = grid.cell(sc.starts[i]), grid.cell(sc.goals[i])
        ax.scatter([scol], [sr], color="lime", marker="o", s=40, edgecolors="black", zorder=5)
        ax.scatter([gc], [gr], color="lime", marker="*", s=110, edgecolors="black", zorder=5)
        d = np.abs(np.asarray(W[i]) - np.asarray(W[order[0]])).mean()
        ax.set_title(f"before agent #{k + 1} (id {i}); mean |w - w_first| {d:.3f}", fontsize=9)


def plot_cost_evolution(grid, sc, W, paths, out_path, title, max_panels=8):
    """How a sequential model's costs change from agent to agent.

    Panels: the cost map used by selected agents in priority order (gradient,
    blue = cheaper, red = dearer than 1), with the paths already reserved in
    black and the current agent's path in green. The last panel shows how much
    each edge's cost varies across all agents (standard deviation).
    """
    import numpy as np
    from matplotlib.colors import Normalize

    n = len(sc.order)
    picks = sorted(set(np.linspace(0, n - 1, min(max_panels - 1, n)).round().astype(int)))
    cols = 4
    rows = int(np.ceil((len(picks) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.6 * rows + 0.5), squeeze=False)
    flat = list(axes.flat)
    norm = weight_norm([W[i] for i in sc.order])
    _evolution_panels(grid, sc, W, paths, picks, flat, norm)
    std = np.std(np.stack(W), axis=0)
    ax = flat[len(picks)]
    vals = draw_weights(ax, grid, std, Normalize(0, max(float(std.max()), 1e-9)), cmap="viridis")
    ax.set_title(f"std of each cost across agents (max {vals.max():.3f})", fontsize=9)
    for a in flat[len(picks) + 1:]:
        a.axis("off")
    fig.subplots_adjust(top=0.9, hspace=0.15, wspace=0.05, left=0.02, right=0.9, bottom=0.02)
    cax = fig.add_axes([0.925, 0.15, 0.012, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="coolwarm"), cax=cax)
    cb.set_label("edge cost (blue cheaper, red dearer than 1)")
    fig.suptitle(title + "\nblack: paths already reserved; green: the agent being planned",
                 fontsize=10)
    return _save(fig, out_path, dpi=110)


def animate_cost_evolution(grid, sc, W, paths, out_path, fps=3):
    """GIF: the cost map before each agent in priority order."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    norm = weight_norm([W[i] for i in sc.order])
    fig, ax = plt.subplots(figsize=(6, 6.3))

    def frame(k):
        ax.clear()
        _evolution_panels(grid, sc, W, paths, [k], [ax], norm)
        return []

    anim = FuncAnimation(fig, frame, frames=len(sc.order), blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
