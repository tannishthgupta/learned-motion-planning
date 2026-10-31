"""
Matplotlib rendering of the world, the paths, and each planner's internals.

This module draws *spatial* things -- the workspace, obstacles, the final
path, and the structure each algorithm built while searching (A*'s expanded
cells, the RRT/RRT* trees). Statistical comparison charts belong to the
benchmark/evaluation stage and live elsewhere.

Showing the internals rather than just the final path is the point: the tree
and the expanded set are what make the algorithmic difference visible, and a
figure of RRT's sprawling tree next to RRT*'s rewired one explains the path
length gap far better than a table.
"""

import matplotlib

# Use a non-interactive backend by default so figures can be generated in a
# plain terminal or CI run with no display attached.
if matplotlib.get_backend().lower() not in ("tkagg", "qtagg", "macosx", "nbagg"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle

from env.obstacles import CircleObstacle, RectObstacle

# One colour per planner, used consistently across every figure in the report --
# the path plots here and the benchmark charts in viz/charts.py share this exact
# mapping, so a reader who learns "RRT* is aqua" once never has to relearn it.
# These are categorical slots 1-3 of the validated palette; see viz/charts.py for
# the measured colour-vision-deficiency separations.
PLANNER_COLORS = {
    "A*": "#2a78d6",      # slot 1, blue
    "RRT": "#eb6834",     # slot 2, orange
    "RRT*": "#1baf7a",    # slot 3, aqua
}

OBSTACLE_FACE = "#4a5568"
OBSTACLE_EDGE = "#2d3748"

# Start and goal are landmarks, not series. They deliberately take status
# colours rather than categorical slots, so a marker can never be mistaken for
# a planner's colour.
START_COLOR = "#0ca30c"
GOAL_COLOR = "#d03b3b"


# ----------------------------------------------------------------------------
# World
# ----------------------------------------------------------------------------

def plot_environment(env, ax=None, show_endpoints=True, title=None):
    """Draw the workspace: obstacles, bounds, start and goal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for o in env.obstacles:
        if isinstance(o, RectObstacle):
            ax.add_patch(Rectangle((o.x, o.y), o.w, o.h, facecolor=OBSTACLE_FACE,
                                   edgecolor=OBSTACLE_EDGE, linewidth=0.8, zorder=2))
        elif isinstance(o, CircleObstacle):
            ax.add_patch(Circle((o.cx, o.cy), o.r, facecolor=OBSTACLE_FACE,
                                edgecolor=OBSTACLE_EDGE, linewidth=0.8, zorder=2))

    if show_endpoints:
        ax.plot(*env.start, marker="o", markersize=9, color=START_COLOR,
                markeredgecolor="white", markeredgewidth=1.4, zorder=6, label="start")
        ax.plot(*env.goal, marker="*", markersize=16, color=GOAL_COLOR,
                markeredgecolor="white", markeredgewidth=1.0, zorder=6, label="goal")

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#a0aec0")
    if title:
        ax.set_title(title, fontsize=11)
    return ax


# ----------------------------------------------------------------------------
# Planner internals
# ----------------------------------------------------------------------------

def plot_tree(ax, tree, color="#888888", linewidth=0.5, alpha=0.6):
    """Draw every edge of an RRT / RRT* tree.

    Uses a LineCollection rather than one plot() call per edge: a converged
    RRT* tree holds thousands of edges, and drawing them individually makes
    figure generation take longer than the planning did.
    """
    if tree is None or len(tree) < 2:
        return ax
    ax.add_collection(LineCollection(tree.edges(), colors=color,
                                     linewidths=linewidth, alpha=alpha, zorder=3))
    return ax


def plot_expanded_cells(ax, grid, expanded, color="#1f77b4", alpha=0.30):
    """Shade the grid cells A* popped from the open list.

    The size of this region is the visual analogue of the `nodes` metric: it
    shows how much of the world A* had to examine before reaching the goal.

    Drawn as an image rather than as markers, so the cells tile exactly at any
    resolution. Scatter markers are sized in points, not data units, so they
    would leave gaps or overlap depending on the figure size.
    """
    if not expanded:
        return ax

    mask = np.zeros((grid.ny, grid.nx), dtype=float)
    for ix, iy in expanded:
        mask[iy, ix] = 1.0

    # Masked cells render fully transparent, leaving only the explored region.
    shaded = np.ma.masked_where(mask == 0.0, mask)
    cmap = ListedColormap([color])

    ax.imshow(shaded, cmap=cmap, alpha=alpha, origin="lower",
              extent=(0, grid.env.width, 0, grid.env.height),
              interpolation="nearest", zorder=3, aspect="equal")
    return ax


def plot_path(ax, path, color="#d62728", linewidth=2.2, label=None,
              show_waypoints=False):
    """Draw a planned path as a polyline."""
    if not path:
        return ax
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, color=color, linewidth=linewidth, solid_capstyle="round",
            zorder=5, label=label)
    if show_waypoints:
        ax.plot(xs, ys, linestyle="none", marker="o", markersize=2.5,
                color=color, zorder=5)
    return ax


def plot_planner_result(env, planner, path, metrics, ax=None, show_internals=True):
    """One panel: the world, the planner's internal structure, and its path.

    Reads whatever internals the planner exposed on its last run, so it works
    for all three without the caller special-casing the algorithm.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    plot_environment(env, ax=ax, show_endpoints=False)
    color = PLANNER_COLORS.get(metrics["planner"], "#333333")

    if show_internals:
        # A* exposes `grid` + `expanded`; RRT and RRT* expose `tree`.
        if getattr(planner, "expanded", None) and getattr(planner, "grid", None):
            plot_expanded_cells(ax, planner.grid, planner.expanded, color=color)
        if getattr(planner, "tree", None) is not None:
            plot_tree(ax, planner.tree, color=color, alpha=0.45)

    plot_path(ax, path, color=color)

    # Re-draw endpoints last so they sit above the tree and path.
    ax.plot(*env.start, marker="o", markersize=9, color=START_COLOR,
            markeredgecolor="white", markeredgewidth=1.4, zorder=7)
    ax.plot(*env.goal, marker="*", markersize=16, color=GOAL_COLOR,
            markeredgecolor="white", markeredgewidth=1.0, zorder=7)

    if metrics["success"]:
        subtitle = "length %.1f   |   %.3f s   |   %d nodes" % (
            metrics["path_length"], metrics["time_sec"], metrics["nodes"])
    else:
        subtitle = "FAILED: %s" % metrics["reason"]

    ax.set_title("%s\n%s" % (metrics["planner"], subtitle), fontsize=10)
    return ax


# ----------------------------------------------------------------------------
# Multi-planner figures
# ----------------------------------------------------------------------------

def compare_planners_figure(env, runs, suptitle=None, figsize=None):
    """Side-by-side panels, one per planner, on the same world.

    `runs` is a list of (planner, path, metrics) tuples.
    """
    n = len(runs)
    if figsize is None:
        figsize = (5.0 * n, 5.6)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (planner, path, metrics) in zip(axes, runs):
        plot_planner_result(env, planner, path, metrics, ax=ax)

    fig.suptitle(suptitle or ("Environment: %s" % env.name), fontsize=13,
                 fontweight="bold")
    # Reserve the top strip for the suptitle; the panel titles are two lines
    # tall, so plain tight_layout() lets them run into it.
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def overlay_paths_figure(env, runs, ax=None, title=None):
    """All planners' paths on a single copy of the world, for direct comparison.

    Stripping out the trees makes the length difference the only visible
    signal, which is what the comparison is actually about.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))

    plot_environment(env, ax=ax)
    for _, path, metrics in runs:
        if not metrics["success"]:
            continue
        plot_path(ax, path, color=PLANNER_COLORS.get(metrics["planner"], "#333"),
                  label="%s (%.1f)" % (metrics["planner"], metrics["path_length"]))

    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)
    ax.set_title(title or ("Path comparison: %s" % env.name), fontsize=12)
    return ax


def save(fig, path, dpi=150):
    """Save and close a figure (closing matters when generating many)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
