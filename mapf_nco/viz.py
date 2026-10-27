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


def plot_comparison(grid, sc, plans, titles, out_path, reference):
    """Side-by-side panels, one per method.

    Agents whose path differs from the reference plan (original PP) are drawn
    thick and in colour; agents with identical paths are faded grey. In the
    reference panel, an agent is highlighted if any other method changed it.
    """
    names = [k for k, v in plans.items() if v is not None]
    changed = changed_agents({k: plans[k] for k in names}, reference)
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 5.6), squeeze=False)
    for ax, name in zip(axes[0], names):
        _draw_map(ax, grid)
        for i, p in enumerate(plans[name]):
            xs, ys = _xy(grid, p)
            if i in changed[name]:
                c = _agent_color(i)
                ax.plot(xs, ys, color=c, lw=2.2, zorder=2)
                ax.scatter(xs[:1], ys[:1], color=c, marker="o", s=30, edgecolors="black", zorder=4)
                ax.scatter(xs[-1:], ys[-1:], color=c, marker="*", s=90, edgecolors="black", zorder=4)
                ax.annotate(str(i), (xs[0], ys[0]), fontsize=7, xytext=(3, 3),
                            textcoords="offset points", zorder=5)
            else:
                ax.plot(xs, ys, color=FADED, lw=1.0, zorder=1)
        ax.set_title(titles[name], fontsize=10)
    fig.suptitle("Coloured = agents whose path differs from the original PP plan; "
                 "grey = unchanged", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_path, dpi=130)


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
                 "Expert": "#41ab5d"}


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
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_path, dpi=130)
