"""
Comparison charts for the benchmark results.

Design notes -- these are deliberate choices, not library defaults:

* **Palette.** The three series use categorical slots 1-3 of a validated
  palette (blue / orange / aqua). The choice was *computed*, not eyeballed:
  across all pairs the worst colour-vision-deficient separation is Delta-E 9.2
  (target >= 8) and the worst normal-vision separation is 24.0 (floor >= 15).
  Aqua measures 2.74:1 against the chart surface, below the 3:1 contrast floor;
  the documented relief is visible direct labels plus a table view, so every
  mark carries its value and the same numbers go to `summary.md`.

* **Colour follows the planner, never its rank.** A* is always blue, RRT always
  orange, RRT* always aqua, in every figure including the path plots.

* **One y-axis per chart, always.** Path length and computation time are never
  drawn against two scales on one plot -- the alignment of two such axes is
  arbitrary and invents a relationship the data does not contain.

* **Form follows the job.**
  - Path length and search effort are magnitude comparisons -> grouped columns.
  - Timing spans three orders of magnitude -> a dumbbell on a log axis. Bars
    grown from a baseline on a log scale would encode an arbitrary origin as
    length. The dumbbell also shows what bars could not: the span from a
    planner's first feasible path to its total runtime, which is precisely the
    anytime behaviour separating RRT* from RRT.
  - Success rate at the full budget is 100% for all three, so a bar chart of it
    would be three identical bars -- a chart where the story is one number.
    Instead the success metric is drawn as a *cumulative success profile*:
    the share of runs solved within a given time budget, swept across budgets.
    That is derived from the recorded first-solution times, not from re-running.

* **Marks thin, chrome recessive.** Hairline solid gridlines on the value axis
  only, no top/right spines, column caps rounded at the data end and square at
  the baseline, a surface-coloured gap between adjacent columns, and a surface
  ring on overlapping markers.

Pixel specs below are in CSS pixels and converted for the output DPI, so a
"24px" column is 24px at the figure's nominal scale rather than 1.7x that.
"""

import os

import matplotlib

if matplotlib.get_backend().lower() not in ("tkagg", "qtagg", "macosx", "nbagg"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# --- Palette -----------------------------------------------------------------
# Categorical slots 1-3 of the validated palette. Never cycled, never reordered.
PLANNER_COLORS = {
    "A*": "#2a78d6",     # slot 1, blue
    "RRT": "#eb6834",    # slot 2, orange
    "RRT*": "#1baf7a",   # slot 3, aqua
}
PLANNER_ORDER = ["A*", "RRT", "RRT*"]

# Chart chrome and ink.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT = ["Segoe UI", "DejaVu Sans", "sans-serif"]

SAVE_DPI = 170
_CSS = SAVE_DPI / 96.0          # CSS px -> device px at the output DPI


def _css_px(n):
    """Convert a CSS-pixel spec into device pixels at the output DPI."""
    return n * _CSS


def _apply_base_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
    })


def _new_figure(width_in, height_in):
    """Create a figure whose DPI matches the output, so px maths is truthful."""
    _apply_base_style()
    fig = plt.figure(figsize=(width_in, height_in), dpi=SAVE_DPI)
    return fig


def _style_axes(ax, value_axis="y", grid=True):
    """Recessive chrome: hairline solid grid on the value axis only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)

    ax.tick_params(length=0, labelsize=9)
    if grid:
        ax.grid(axis=value_axis, color=GRIDLINE, linewidth=1.0, linestyle="-",
                zorder=0)
        ax.set_axisbelow(True)
    return ax


def _header(fig, title, subtitle, planners=None):
    """Stack title, subtitle and legend in figure coordinates.

    All three live on the figure rather than the axes, and each is placed
    relative to the block above it, so adding a line to the subtitle pushes the
    legend down instead of letting them overlap. Axes-relative padding cannot
    do this -- it shifts whenever the tick labels change width.

    Returns the y coordinate where the plotting area may begin.
    """
    fig.text(0.015, 0.975, title, fontsize=13, fontweight="bold",
             color=INK_PRIMARY, va="top", ha="left")
    fig.text(0.015, 0.900, subtitle, fontsize=9.5, color=INK_SECONDARY,
             va="top", ha="left", linespacing=1.45)

    n_lines = subtitle.count(chr(10)) + 1
    y = 0.900 - 0.050 * n_lines

    if planners:
        # A legend is always present for two or more series.
        handles = [Line2D([], [], marker="o", linestyle="none", markersize=7,
                          markerfacecolor=PLANNER_COLORS[p],
                          markeredgecolor="none", label=p) for p in planners]
        leg = fig.legend(handles=handles, loc="upper left",
                         bbox_to_anchor=(0.012, y), ncol=len(planners),
                         frameon=False, fontsize=9, handletextpad=0.4,
                         columnspacing=1.5, borderaxespad=0.0)
        for text in leg.get_texts():
            text.set_color(INK_SECONDARY)
        y -= 0.055

    return y


def _add_axes(fig, title, subtitle, planners=None, left=0.10, bottom=0.14,
              width=0.86):
    """Draw the header block, then return an axes sized to fit beneath it."""
    top = _header(fig, title, subtitle, planners)
    return fig.add_axes([left, bottom, width, max(0.30, top - bottom - 0.02)])


def _px_to_data(ax, px):
    """Convert a device-pixel length into x and y data-unit lengths.

    Needed so a rounded column corner renders as an actual circle: the x and y
    data scales differ, so one data-space radius would come out elliptical.
    Call only after the axis limits are final.
    """
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0.0, 0.0))
    x1, y1 = inv.transform((px, px))
    return abs(x1 - x0), abs(y1 - y0)


def _rounded_column(ax, x_left, width, height, color, radius_css=4.0, baseline=0.0):
    """Draw a column: square at the baseline, rounded at the data end."""
    rx, ry = _px_to_data(ax, _css_px(radius_css))
    rx = min(rx, width / 2.0)
    ry = min(ry, abs(height) / 2.0) if height else 0.0

    left, right = x_left, x_left + width
    top = baseline + height

    if ry <= 0 or rx <= 0:
        verts = [(left, baseline), (left, top), (right, top), (right, baseline),
                 (left, baseline)]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    else:
        verts = [
            (left, baseline),
            (left, top - ry),
            (left, top), (left + rx, top),
            (right - rx, top),
            (right, top), (right, top - ry),
            (right, baseline),
            (left, baseline),
        ]
        codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
                 Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
                 Path.CLOSEPOLY]

    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none",
                           zorder=3))


def _densities(summary):
    """Density labels in natural order, excluding the pooled 'all' row."""
    seen = [s["density_label"] for s in summary]
    return [d for d in dict.fromkeys(seen) if d != "all"]


def _lookup(summary, planner, density, field):
    for s in summary:
        if s["planner"] == planner and s["density_label"] == density:
            return s[field]
    return float("nan")


# ----------------------------------------------------------------------------
# Grouped columns (magnitude comparisons)
# ----------------------------------------------------------------------------

def _grouped_columns(ax, summary, field, densities, planners, value_fmt="%.0f",
                     err_field=None, headroom=1.16):
    """Shared layout for the grouped column charts.

    Column thickness is capped at 24 CSS px and a 2px surface gap separates
    adjacent columns, so the group reads as distinct marks without any stroke
    being drawn around them.
    """
    n_groups, n_series = len(densities), len(planners)
    ax.set_xlim(-0.5, n_groups - 0.5)

    values = [[_lookup(summary, p, d, field) for d in densities] for p in planners]
    errs = ([[_lookup(summary, p, d, err_field) for d in densities] for p in planners]
            if err_field else None)

    top = np.nanmax([v for row in values for v in row])
    if errs:
        top = np.nanmax([v + (e if np.isfinite(e) else 0.0)
                         for row, erow in zip(values, errs)
                         for v, e in zip(row, erow)])
    ax.set_ylim(0, top * headroom)

    gap_x, _ = _px_to_data(ax, _css_px(2.0))
    max_bar_x, _ = _px_to_data(ax, _css_px(24.0))

    slot = 0.74 / n_series
    bar_w = min(slot - gap_x, max_bar_x)
    group_w = n_series * bar_w + (n_series - 1) * gap_x

    for si, planner in enumerate(planners):
        color = PLANNER_COLORS[planner]
        for gi in range(n_groups):
            value = values[si][gi]
            if not np.isfinite(value):
                continue
            left = gi - group_w / 2.0 + si * (bar_w + gap_x)
            _rounded_column(ax, left, bar_w, value, color)

            label_y = value
            if errs and np.isfinite(errs[si][gi]) and errs[si][gi] > 0:
                err = errs[si][gi]
                ax.plot([left + bar_w / 2.0] * 2, [value - err, value + err],
                        color=INK_MUTED, linewidth=1.0, zorder=4,
                        solid_capstyle="butt")
                label_y = value + err

            # Direct value label on the cap -- mandatory here, since one series
            # sits below the contrast floor and labels are the documented relief.
            ax.annotate(value_fmt % value, xy=(left + bar_w / 2.0, label_y),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5,
                        color=INK_SECONDARY)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([d.capitalize() for d in densities], fontsize=9.5,
                       color=INK_SECONDARY)
    return ax


def chart_path_length(summary, out_path, densities=None, planners=None):
    """Mean path length on the common-solved subset -- the quality metric."""
    planners = planners or PLANNER_ORDER
    densities = densities or _densities(summary)

    fig = _new_figure(7.4, 4.8)
    ax = _add_axes(fig, "Path length by obstacle density",
                   "Mean over instances all three planners solved; whiskers are "
                   "±1 s.d. Lower is better.",
                   planners, left=0.10, bottom=0.13, width=0.87)
    _grouped_columns(ax, summary, "length_mean_common", densities, planners,
                     "%.0f", err_field="length_std_common")
    _style_axes(ax)
    ax.set_ylabel("Path length (world units)", fontsize=9.5, color=INK_SECONDARY)
    return _save(fig, out_path)


def chart_nodes(summary, out_path, densities=None, planners=None):
    """Search effort: cells expanded (A*) or tree nodes (RRT / RRT*)."""
    planners = planners or PLANNER_ORDER
    densities = densities or _densities(summary)

    fig = _new_figure(7.4, 4.8)
    ax = _add_axes(fig, "Search effort by obstacle density",
                   "A*: grid cells expanded.    RRT / RRT*: nodes in the final tree.",
                   planners, left=0.11, bottom=0.13, width=0.86)
    _grouped_columns(ax, summary, "nodes_mean", densities, planners, "%.0f")
    _style_axes(ax)
    ax.set_ylabel("Nodes", fontsize=9.5, color=INK_SECONDARY)
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# Success: cumulative profile over the time budget
# ----------------------------------------------------------------------------

def chart_success_profile(rows, out_path, planners=None, n_points=240):
    """Share of runs solved within a given time budget, swept across budgets.

    Success rate is only a discriminating metric while the budget actually
    binds. At the full 10 s budget all three planners reach 100%, so a bar
    chart of that single number would say nothing. Sweeping the budget shows
    *when* each planner becomes reliable, which is the real difference.

    Computed from the recorded first-solution times: a run that first found a
    path at time t counts as a success for every budget >= t. No re-running is
    involved, so this is a view of the same data, not a separate experiment.
    """
    planners = planners or PLANNER_ORDER

    times = {}
    for planner in planners:
        subset = [r for r in rows if r["planner"] == planner]
        # Failures are recorded as infinite time: never solved at any budget.
        times[planner] = (np.array([r["first_solution_time"] if r["success"]
                                    else np.inf for r in subset]), len(subset))

    finite = np.concatenate([t[np.isfinite(t)] for t, _ in times.values()])
    lo, hi = finite.min() * 0.7, finite.max() * 1.6
    budgets = np.geomspace(lo, hi, n_points)

    fig = _new_figure(7.6, 4.9)
    ax = _add_axes(fig, "How quickly each planner becomes reliable",
                   "Share of runs that had found a valid path within a given "
                   "budget. All three reach 100% - the difference is when." + chr(10) +
                   "Dots mark the median run.",
                   planners, left=0.09, bottom=0.14, width=0.84)

    for planner in planners:
        t, n = times[planner]
        pct = [100.0 * np.count_nonzero(t <= b) / n for b in budgets]
        ax.plot(budgets, pct, color=PLANNER_COLORS[planner], linewidth=2.0,
                solid_capstyle="round", zorder=4)

    # Direct labels at each median-crossing. Curves whose medians sit close
    # together on the log axis would overlap, so labels are lifted in turn
    # rather than stacked on one line -- nudging them apart horizontally would
    # detach each label from the dot it belongs to.
    medians = sorted(((float(np.median(times[p][0][np.isfinite(times[p][0])])), p)
                      for p in planners), key=lambda kv: kv[0])
    last_x, level = None, 0
    for median_t, planner in medians:
        if last_x is not None and abs(np.log10(median_t) - np.log10(last_x)) < 0.3:
            level = 1 - level
        else:
            level = 0
        last_x = median_t

        ax.plot(median_t, 50.0, marker="o", markersize=8,
                markerfacecolor=PLANNER_COLORS[planner], markeredgecolor=SURFACE,
                markeredgewidth=2.0, zorder=5)
        ax.annotate("%s  %s s" % (planner, _short_time(median_t)),
                    xy=(median_t, 50.0), xytext=(0, 14 + 18 * level),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK_SECONDARY)

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-3, 106)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    _style_axes(ax)
    ax.set_xlabel("Time budget (seconds, log scale)", fontsize=9.5,
                  color=INK_SECONDARY)
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# Timing: dumbbell on a log axis
# ----------------------------------------------------------------------------

def _short_time(seconds):
    if seconds < 0.01:
        return "%.4f" % seconds
    if seconds < 1:
        return "%.3f" % seconds
    return "%.2f" % seconds


def chart_time(summary, out_path, densities=None, planners=None):
    """Computation time, as a span from first feasible path to total runtime.

    A log axis is required (three orders of magnitude), and columns on a log
    axis would encode an arbitrary origin as length -- so this is a dumbbell
    plot. For A* and RRT the two ends coincide, because both stop as soon as
    they have a path. For RRT* the span has visible length, and that length
    *is* the refinement time.
    """
    planners = planners or PLANNER_ORDER
    densities = densities or _densities(summary)

    fig = _new_figure(7.8, 5.0)
    ax = _add_axes(fig, "Computation time by obstacle density",
                   "Filled dot: first feasible path.    Hollow dot: total runtime." + chr(10) +
                   "The span between them is RRT*'s refinement, which RRT and A* "
                   "do not do.",
                   planners, left=0.17, bottom=0.14, width=0.72)

    y_positions, y = {}, 0.0
    for density in densities:
        for planner in planners:
            y_positions[(density, planner)] = y
            y -= 1.0
        y -= 0.7

    spans = {}
    for key in y_positions:
        density, planner = key
        first = _lookup(summary, planner, density, "first_time_median")
        total = _lookup(summary, planner, density, "time_median")
        if np.isfinite(first) and np.isfinite(total):
            spans[key] = (first, total)

    xmin = min(f for f, _ in spans.values())
    xmax = max(t for _, t in spans.values())
    ax.set_xscale("log")
    ax.set_xlim(xmin * 0.45, xmax * 4.0)
    ax.set_ylim(min(y_positions.values()) - 0.8, max(y_positions.values()) + 0.8)

    for (density, planner), (first, total) in spans.items():
        ypos = y_positions[(density, planner)]
        color = PLANNER_COLORS[planner]

        if total > first * 1.05:
            ax.plot([first, total], [ypos, ypos], color=color, linewidth=2.0,
                    solid_capstyle="round", alpha=0.45, zorder=3)
            ax.plot(total, ypos, marker="o", markersize=8, markerfacecolor=SURFACE,
                    markeredgecolor=color, markeredgewidth=2.0, zorder=4)
            ax.annotate("%s s" % _short_time(total), xy=(total, ypos), xytext=(10, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=INK_SECONDARY)
        else:
            ax.annotate("%s s" % _short_time(first), xy=(first, ypos), xytext=(10, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=INK_SECONDARY)

        # Surface ring keeps the marker legible where it overlaps the span.
        ax.plot(first, ypos, marker="o", markersize=8, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=5)

    ax.set_yticks([y_positions[(d, p)] for d in densities for p in planners])
    ax.set_yticklabels([p for _ in densities for p in planners], fontsize=9,
                       color=INK_SECONDARY)

    # Density group labels, set clear of the planner tick labels.
    for density in densities:
        ys = [y_positions[(density, p)] for p in planners]
        ax.annotate(density.capitalize(), xy=(0, float(np.mean(ys))),
                    xycoords=("axes fraction", "data"), xytext=(-86, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=9.5, color=INK_PRIMARY, fontweight="bold")

    _style_axes(ax, value_axis="x")
    ax.set_xlabel("Median planning time (seconds, log scale)", fontsize=9.5,
                  color=INK_SECONDARY)
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# Convergence: small multiples, one panel per density
# ----------------------------------------------------------------------------

def _resample_history(history, grid):
    """Step-interpolate one RRT* cost curve onto a shared iteration grid.

    The curve is a step function -- the best cost holds until the next
    improvement. Iterations before that run's first solution are NaN.
    """
    iters = np.array([h[0] for h in history], dtype=float)
    costs = np.array([h[2] for h in history], dtype=float)
    out = np.full(len(grid), np.nan)
    idx = np.searchsorted(iters, grid, side="right") - 1
    valid = idx >= 0
    out[valid] = costs[idx[valid]]
    return out


def chart_convergence(convergence, summary, out_path, densities=None,
                      max_iters=None):
    """RRT*'s anytime refinement, against A* and RRT as reference levels.

    Faceted by density rather than pooled: absolute path length differs by
    density, so one pooled plot would blur three different scales together.

    A run that has not yet found its first path is scored as *infinitely* bad
    rather than dropped. That keeps the population fixed at every iteration,
    which matters more than it sounds: averaging only over the runs that
    already have a path silently changes the population as the plot moves
    right, and because the early finishers are the lucky short-path ones, the
    "median" then *rises* even though no individual run ever gets worse.

    With unsolved runs held at infinity, each run's own curve is monotone
    non-increasing, so the median of the fixed set is too. The median becomes
    finite once half the runs have a path, and the band once three-quarters do
    -- which is exactly where each is meaningful. Percentiles are taken by
    nearest rank, so no interpolation against infinity is ever performed.
    """
    densities = densities or _densities(summary)
    if not convergence:
        return None

    if max_iters is None:
        max_iters = max(h[0] for c in convergence for h in c["history"])
    grid = np.unique(np.round(np.geomspace(1, max_iters, 260)).astype(int))

    fig = _new_figure(4.9 * len(densities), 4.9)
    _header(fig, "RRT* keeps improving after it finds its first path",
            "Median best-known path length vs iteration; the band is the "
            "interquartile range across runs. Horizontal lines are the other "
            "planners' mean length" + chr(10) + "on the same worlds. A run with "
            "no path yet counts as infinitely long, so each curve starts where "
            "half the runs have one.")
    axes = [fig.add_axes([0.06 + i * (0.92 / len(densities)), 0.12,
                          0.92 / len(densities) - 0.075, 0.60])
            for i in range(len(densities))]

    for ax, density in zip(axes, densities):
        curves = [c for c in convergence if c["density_label"] == density]
        if not curves:
            continue

        stack = np.vstack([_resample_history(c["history"], grid) for c in curves])
        # Unsolved-so-far becomes +inf, keeping the population size fixed.
        filled = np.where(np.isfinite(stack), stack, np.inf)
        ordered = np.sort(filled, axis=0)
        n = ordered.shape[0]

        def _rank(q):
            return ordered[min(n - 1, max(0, int(np.ceil(q * n)) - 1))]

        median, q1, q3 = _rank(0.50), _rank(0.25), _rank(0.75)

        star = PLANNER_COLORS["RRT*"]
        band = np.isfinite(q3)
        if band.any():
            ax.fill_between(grid[band], q1[band], q3[band], color=star, alpha=0.10,
                            linewidth=0, zorder=2)
        show = np.isfinite(median)
        gx, median = grid[show], median[show]
        ax.plot(gx, median, color=star, linewidth=2.0, solid_capstyle="round",
                zorder=4)

        for planner in ("A*", "RRT"):
            level = _lookup(summary, planner, density, "length_mean_common")
            if not np.isfinite(level):
                continue
            ax.axhline(level, color=PLANNER_COLORS[planner], linewidth=1.6,
                       alpha=0.9, zorder=3)
            ax.annotate(planner, xy=(1.0, level), xycoords=("axes fraction", "data"),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK_SECONDARY)

        if len(median):
            ax.annotate("RRT*", xy=(1.0, median[-1]),
                        xycoords=("axes fraction", "data"), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=INK_SECONDARY, fontweight="bold")

        ax.set_xscale("log")
        _style_axes(ax)
        ax.set_title(density.capitalize(), fontsize=10.5, loc="left",
                     color=INK_PRIMARY, pad=6)
        ax.set_xlabel("Iterations (log scale)", fontsize=9, color=INK_SECONDARY)

    axes[0].set_ylabel("Best path length so far", fontsize=9.5, color=INK_SECONDARY)
    return _save(fig, out_path)


# ----------------------------------------------------------------------------

def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def make_all_charts(summary, rows, convergence, out_dir="results", suffix=""):
    """Generate every benchmark figure. Returns the list of written paths.

    `suffix` keeps a throwaway (--quick) run from overwriting the figures the
    report already cites.
    """
    def _out(name):
        return os.path.join(out_dir, "chart_%s%s.png" % (name, suffix))

    written = [
        chart_success_profile(rows, _out("success_profile")),
        chart_path_length(summary, _out("path_length")),
        chart_time(summary, _out("time")),
        chart_nodes(summary, _out("nodes")),
    ]
    conv = chart_convergence(convergence, summary, _out("convergence"))
    if conv:
        written.append(conv)
    return written
