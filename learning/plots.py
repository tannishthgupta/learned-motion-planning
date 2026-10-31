"""
Figures for the learned-sampling extension.

    python -m learning.plots

Reads only saved results (results/learned/*.csv / *.json) plus the saved
model; it never re-runs the evaluation, so figures always match the tables.

    learned_convergence.png   normalised cost vs iteration, per test set
    learned_heatmaps.png      scenarios: exact region, prediction, both trees
    learned_training.png      loss and IoU per epoch

Styling reuses the project's chart helpers (viz/charts.py) so the figures
match the rest of the report. RRT* keeps its aqua; the learned variant is
violet (palette slot 7; the pair passes the colour-vision checks); the exact
Dijkstra sampler is drawn as a muted dashed reference line because it is a
reference point, not a competing method.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from env import SCENARIOS, get_scenario
from planners import RRTStarPlanner
from viz.charts import (INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE, _header,
                        _new_figure, _save, _style_axes)
from viz.plot import PLANNER_COLORS, plot_environment, plot_path, plot_tree

from .evaluate import COMMON, MODEL_PATH, OUT_DIR, best_known, load_rows

LEARNED = "#6250d6"
METHOD_STYLE = {
    "RRT*": dict(color=PLANNER_COLORS["RRT*"], linestyle="-", label="RRT* (uniform)"),
    "RRT* + learned": dict(color=LEARNED, linestyle="-", label="RRT* + learned"),
    "RRT* + Dijkstra": dict(color=INK_MUTED, linestyle=(0, (4, 3)),
                            label="RRT* + exact Dijkstra (reference)"),
}
SET_TITLES = {"test": "Held-out random problems",
              "benchmark": "Main benchmark worlds",
              "scenarios": "Hand-designed scenarios (unseen structure)"}

# Sequential ramp for the heatmaps: one hue, surface -> violet.
HEAT = LinearSegmentedColormap.from_list("violet", [SURFACE, "#b9b1f0", LEARNED])


def _rank(sorted_stack, q):
    n = sorted_stack.shape[0]
    return sorted_stack[min(n - 1, max(0, int(np.ceil(q * n)) - 1))]


def _curve(history, grid):
    """Step-function best cost at each grid iteration (inf before first path)."""
    out = np.full(len(grid), np.inf)
    if not history:
        return out
    it = np.array([h[0] for h in history], dtype=float)
    c = np.minimum.accumulate(np.array([h[1] for h in history], dtype=float))
    idx = np.searchsorted(it, grid, side="right") - 1
    ok = idx >= 0
    out[ok] = c[idx[ok]]
    return out


def chart_convergence(rows, max_iters, out_path):
    ref = best_known(rows)
    grid = np.unique(np.round(np.geomspace(50, max_iters, 200)).astype(int))
    sets = [s for s in ("test", "benchmark", "scenarios") if any(r["set"] == s for r in rows)]

    fig = _new_figure(4.6 * len(sets), 4.9)
    _header(fig, "Learned sampling makes RRT* converge in fewer iterations",
            "Median normalised path cost (1.00 = best cost any method found on that problem) "
            "vs iteration; bands are the interquartile range." + chr(10) +
            "A run with no path yet counts as infinitely long, so each curve starts "
            "once half the runs have a path.")
    axes = [fig.add_axes([0.06 + i * (0.93 / len(sets)), 0.13, 0.93 / len(sets) - 0.085, 0.53])
            for i in range(len(sets))]

    for ax, set_name in zip(axes, sets):
        ends = []
        for method, style in METHOD_STYLE.items():
            rs = [r for r in rows if r["set"] == set_name and r["method"] == method]
            if not rs:
                continue
            stack = np.vstack([_curve(r["history"], grid) / ref.get((r["set"], r["problem"]), np.inf)
                               for r in rs])
            ordered = np.sort(stack, axis=0)
            med, q1, q3 = _rank(ordered, .5), _rank(ordered, .25), _rank(ordered, .75)
            band = np.isfinite(q3)
            if band.any() and method != "RRT* + Dijkstra":
                ax.fill_between(grid[band], q1[band], q3[band], color=style["color"],
                                alpha=0.10, linewidth=0, zorder=2)
            show = np.isfinite(med)
            ax.plot(grid[show], med[show], color=style["color"], linestyle=style["linestyle"],
                    linewidth=2.0, solid_capstyle="round", zorder=4)
            if show.any():
                ends.append((med[show][-1], method))
        # Direct labels at the right edge, nudged apart so they never collide.
        ends.sort()
        last = -np.inf
        span = 0.0
        ax.set_xscale("log")
        ax.set_xlim(grid[0], grid[-1])
        lo, hi = ax.get_ylim()
        hi = min(hi, 1.6)
        ax.set_ylim(0.98, hi)
        span = (hi - 0.98) * 0.07
        short = {"RRT*": "uniform", "RRT* + learned": "learned", "RRT* + Dijkstra": "Dijkstra"}
        for y, method in ends:
            y_lab = max(y, last + span)
            last = y_lab
            ax.annotate(short[method], xy=(1.0, y_lab), xycoords=("axes fraction", "data"),
                        xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5,
                        color=INK_SECONDARY,
                        fontweight="bold" if method == "RRT* + learned" else "normal")
        _style_axes(ax)
        ax.set_title(SET_TITLES[set_name], fontsize=10, loc="left", color=INK_PRIMARY, pad=6)
        ax.set_xlabel("Iterations (log scale)", fontsize=9, color=INK_SECONDARY)
    axes[0].set_ylabel("Normalised cost (median)", fontsize=9.5, color=INK_SECONDARY)

    # Legend (>= 2 series): line swatches, placed under the subtitle.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=s["color"], linestyle=s["linestyle"], linewidth=2,
                      label=s["label"]) for s in METHOD_STYLE.values()]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.80),
                     ncol=3, frameon=False, fontsize=9, columnspacing=1.8)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    return _save(fig, out_path)


def figure_heatmaps(mix, out_path, iters=1000, seed=0, model_path=MODEL_PATH):
    """One row per scenario: exact region | prediction | uniform tree | learned tree."""
    import torch
    torch.set_num_threads(4)
    from .model import load_model
    from .sampler import DijkstraSampler, LearnedSampler
    model, _ = load_model(model_path)

    # The five scenarios, plus the one benchmark world where the learned
    # sampler failed (its only passage is narrower than one grid cell).
    from env import generate_suite
    envs = [get_scenario(n) for n in SCENARIOS]
    fail = [e for e in generate_suite(n_per_density=10, seed=0) if e.name == "cluttered_02"]
    for e in fail:
        e.name = "cluttered_02 (failure case)"
    envs += fail
    names = [e.name for e in envs]
    fig = _new_figure(12.0, 3.0 * len(names) + 1.15)
    fig.text(0.015, 0.995, "What the network predicts, and what it does to the tree",
             fontsize=13, fontweight="bold", color=INK_PRIMARY, va="top")
    fig.text(0.015, 0.995 - 0.55 / fig.get_figheight(),
             "Scenarios were never seen in training. The last row is the one benchmark world where the learned sampler failed "
             "(3 of 5 runs): its only passage is narrower" + chr(10) + "than one grid cell, so the exact region is empty "
             "and the prediction is a guess. Columns: exact near-optimal region (training target) | network prediction | "
             + chr(10) + "RRT* tree after %d iterations with uniform vs learned sampling (same seed)." % iters,
             fontsize=9, color=INK_SECONDARY, va="top")
    top = 1 - 1.15 / fig.get_figheight()
    h = top / len(names)
    cols = ["Exact region (Dijkstra)", "Predicted", "RRT* uniform", "RRT* + learned"]

    for r, (name, env) in enumerate(zip(names, envs)):
        exact = DijkstraSampler(mix=mix)
        exact.prepare(env, env.start, env.goal)
        learned = LearnedSampler(model, mix=mix)
        learned.prepare(env, env.start, env.goal)
        runs = {}
        for key, sampler in (("uniform", None), ("learned", LearnedSampler(model, mix=mix))):
            p = RRTStarPlanner(max_iters=iters, seed=seed, sampler=sampler, **COMMON)
            path, m = p.plan(env)
            runs[key] = (p, path, m)

        for c in range(4):
            ax = fig.add_axes([0.01 + c * 0.247, top - (r + 1) * h + 0.01, 0.235, h - 0.035])
            if c < 2:
                heat = exact.last_probabilities if c == 0 else learned.last_probabilities
                ax.imshow(heat, origin="lower", extent=(0, env.width, 0, env.height),
                          cmap=HEAT, vmin=0, vmax=1, interpolation="nearest", zorder=1)
                plot_environment(env, ax)
                sub = ""
            else:
                p, path, m = runs["uniform" if c == 2 else "learned"]
                color = PLANNER_COLORS["RRT*"] if c == 2 else LEARNED
                plot_environment(env, ax)
                plot_tree(ax, p.tree, color=color, linewidth=0.45, alpha=0.55)
                plot_path(ax, path, color=INK_PRIMARY, linewidth=1.8)
                sub = ("  cost %.1f" % m["path_length"]) if m["success"] else "  no path yet"
            title = (name + " - " if c == 0 else "") + cols[c] + sub
            ax.set_title(title, fontsize=8.5, loc="left", color=INK_SECONDARY, pad=3)
    return _save(fig, out_path)


def chart_training(log_path, out_path):
    log = json.load(open(log_path))
    ep = [e["epoch"] for e in log["epochs"]]
    fig = _new_figure(9.0, 3.8)
    _header(fig, "Training curves",
            "%d training / %d validation maps, %d parameters. Left: masked BCE loss. "
            "Right: validation IoU of the predicted region." % (
                log["n_train"], log["n_val"], log["params"]))
    a1 = fig.add_axes([0.07, 0.15, 0.40, 0.58])
    a2 = fig.add_axes([0.57, 0.15, 0.40, 0.58])
    a1.plot(ep, [e["train_loss"] for e in log["epochs"]], color=LEARNED, linewidth=2)
    a1.plot(ep, [e["val_loss"] for e in log["epochs"]], color=INK_MUTED, linewidth=2,
            linestyle=(0, (4, 3)))
    a1.annotate("train", xy=(ep[-1], log["epochs"][-1]["train_loss"]), xytext=(5, -6),
                textcoords="offset points", fontsize=8.5, color=INK_SECONDARY)
    a1.annotate("validation", xy=(ep[-1], log["epochs"][-1]["val_loss"]), xytext=(5, 6),
                textcoords="offset points", fontsize=8.5, color=INK_SECONDARY)
    a2.plot(ep, [e["val_iou"] for e in log["epochs"]], color=LEARNED, linewidth=2)
    for ax, lab in ((a1, "Loss"), (a2, "Validation IoU")):
        _style_axes(ax)
        ax.set_xlabel("Epoch", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel(lab, fontsize=9, color=INK_SECONDARY)
    return _save(fig, out_path)


def main():
    meta = json.load(open(os.path.join(OUT_DIR, "eval_meta.json")))
    rows = load_rows(os.path.join(OUT_DIR, "eval_raw.csv"),
                     os.path.join(OUT_DIR, "eval_histories.json"))
    for p in (chart_convergence(rows, meta["max_iters"],
                                os.path.join(OUT_DIR, "learned_convergence.png")),
              figure_heatmaps(meta["mix"], os.path.join(OUT_DIR, "learned_heatmaps.png")),
              chart_training(os.path.join(OUT_DIR, "training_log.json"),
                             os.path.join(OUT_DIR, "learned_training.png"))):
        print("  ->", p)


if __name__ == "__main__":
    main()
