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
    "mean": r"$L(\hat y, y^*) = \frac{1}{M}\sum_{e} (\hat y_e - y^*_e)^2$",
    "sum": r"$L(\hat y, y^*) = \sum_{e} (\hat y_e - y^*_e)^2$",
}


def plot_training_curves(history, batch_losses, refs, best_epoch, kind, n, reduction, out_path):
    """Loss and validation delay per epoch, with the loss function in the title."""
    epochs = [h["epoch"] for h in history]
    loss = [h.get("train_loss") for h in history]
    val = [h["val_delay"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))

    if batch_losses:
        bx, by = zip(*batch_losses)
        a1.plot(bx, by, color="#9ecae1", lw=0.8, label="per mini-batch")
    pts = [(e, v) for e, v in zip(epochs, loss) if v is not None]
    if pts:
        a1.plot(*zip(*pts), "o-", color="#08519c", lw=2, label="epoch mean")
    a1.set_xlabel("epoch")
    a1.set_ylabel("usage loss")
    a1.set_title("Training loss   " + LOSS_FORMULA.get(reduction, ""), fontsize=11)
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)

    a2.plot(epochs, val, "o-", color="#d94801", lw=2, label=f"PP ({kind})")
    a2.axhline(refs["orig_pp"], color="gray", ls="--", label=f"original PP ({refs['orig_pp']:.2f})")
    a2.axhline(refs["expert"], color="green", ls=":", label=f"{refs.get('expert_name', 'EECBS')} ({refs['expert']:.2f})")
    a2.scatter([best_epoch], [val[epochs.index(best_epoch)]], s=160, marker="*", color="gold",
               edgecolors="black", zorder=5, label=f"selected (epoch {best_epoch})")
    a2.set_xlabel("epoch")
    a2.set_ylabel(r"validation delay  ($\sum_e y_e$ $-$ lower bound)")
    a2.set_title("Validation delay (lower is better)", fontsize=11)
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    fig.suptitle(f"{kind} model, {n} agents", fontsize=12)
    fig.tight_layout()
    return _save(fig, out_path, dpi=130)


METHOD_COLORS = {"PP (orig)": "#969696", "PP (Edge)": "#4292c6", "PP (GNN)": "#ec7014",
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
