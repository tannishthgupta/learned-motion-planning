"""
Aggregation of raw benchmark rows into the tables the report quotes.

One methodological point drives the whole module.

**Path length may only be averaged over instances every planner solved.**
If planner X fails on the hard worlds and planner Y solves them, then
averaging each planner over "its own successes" flatters X: it is scored only
on the easy instances. The fix is a *common-solved subset* -- the
(world, trial) pairs where all three succeeded -- and quality comparisons are
made there. Success rate is reported separately, over everything, which is
where the failures properly show up.

Both numbers are computed and written out, so the report can show the naive
per-planner mean alongside the like-for-like one.
"""

import csv
import os
from collections import defaultdict

import numpy as np

PLANNER_ORDER = ["A*", "RRT", "RRT*"]

SUMMARY_FIELDS = [
    "planner", "density_label", "density_target",
    "n_runs", "n_success", "success_rate",
    "length_mean", "length_std",                 # over that planner's own successes
    "length_mean_common", "length_std_common",   # over the common-solved subset
    "length_norm_mean",                          # length / best-on-that-instance
    "time_mean", "time_median",                  # total planning time
    "first_time_mean", "first_time_median",      # time to FIRST feasible path
    "nodes_mean", "motion_checks_mean",
]


def _mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def _std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _median(values):
    return float(np.median(values)) if len(values) else float("nan")


def instance_key(row):
    """A single planning problem: one world, one trial index."""
    return (row["environment"], row["trial"])


def common_solved_instances(rows, planners=None):
    """Instances solved by *every* planner -- the fair ground for comparing quality."""
    planners = planners or PLANNER_ORDER
    solved = defaultdict(set)
    for row in rows:
        if row["success"]:
            solved[instance_key(row)].add(row["planner"])
    return {k for k, names in solved.items() if set(planners).issubset(names)}


def best_length_per_instance(rows):
    """Shortest path any planner achieved on each instance.

    Used as the normalisation baseline. Taking the best *observed* result
    rather than A*'s avoids privileging one algorithm as "the reference"
    when the true continuous optimum is unknown.
    """
    best = {}
    for row in rows:
        if not row["success"]:
            continue
        k = instance_key(row)
        if k not in best or row["path_length"] < best[k]:
            best[k] = row["path_length"]
    return best


def summarise(rows, planners=None):
    """Aggregate raw rows into one summary record per (planner, density).

    Also emits an "all" density row per planner, for the headline table.
    """
    planners = planners or PLANNER_ORDER
    common = common_solved_instances(rows, planners)
    best = best_length_per_instance(rows)

    densities = []
    for row in rows:
        if row["density_label"] not in densities:
            densities.append(row["density_label"])

    out = []
    for planner in planners:
        for density in densities + ["all"]:
            subset = [r for r in rows
                      if r["planner"] == planner
                      and (density == "all" or r["density_label"] == density)]
            if not subset:
                continue

            successes = [r for r in subset if r["success"]]
            in_common = [r for r in successes if instance_key(r) in common]

            lengths = [r["path_length"] for r in successes]
            lengths_common = [r["path_length"] for r in in_common]
            norm = [r["path_length"] / best[instance_key(r)] for r in in_common
                    if best.get(instance_key(r))]

            out.append({
                "planner": planner,
                "density_label": density,
                "density_target": (float("nan") if density == "all"
                                   else subset[0]["density_target"]),
                "n_runs": len(subset),
                "n_success": len(successes),
                "success_rate": 100.0 * len(successes) / len(subset),
                "length_mean": _mean(lengths),
                "length_std": _std(lengths),
                "length_mean_common": _mean(lengths_common),
                "length_std_common": _std(lengths_common),
                "length_norm_mean": _mean(norm),
                # Time is summarised by median as well as mean: planning times
                # are right-skewed (a few hard worlds dominate the mean).
                "time_mean": _mean([r["time_sec"] for r in subset]),
                "time_median": _median([r["time_sec"] for r in subset]),
                "first_time_mean": _mean([r["first_solution_time"] for r in successes]),
                "first_time_median": _median([r["first_solution_time"] for r in successes]),
                "nodes_mean": _mean([r["nodes"] for r in subset]),
                "motion_checks_mean": _mean([r["motion_checks"] for r in subset]),
            })
    return out


def summary_by(summary, density):
    """Index a summary by planner for one density label."""
    return {s["planner"]: s for s in summary if s["density_label"] == density}


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def write_summary_csv(summary, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row[k] for k in SUMMARY_FIELDS})
    return path


def _fmt(value, spec="%.2f", nan="-"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return nan
    return spec % value


def markdown_tables(summary, rows, densities=None):
    """Render the report tables as Markdown.

    This is also the chart's required *table view*: the validated palette
    leaves one series below the 3:1 contrast floor, and the documented relief
    for that is visible labels plus a table -- both of which are shipped.
    """
    if densities is None:
        densities = [s["density_label"] for s in summary]
        densities = [d for d in dict.fromkeys(densities) if d != "all"] + ["all"]

    lines = []

    lines.append("### Headline comparison (all densities pooled)\n")
    lines.append("| planner | success rate | path length (common subset) | "
                 "normalised length | time to first path | total time | nodes |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in [s for s in summary if s["density_label"] == "all"]:
        lines.append("| %s | %s%% | %s ± %s | %s× | %s s | %s s | %s |" % (
            s["planner"], _fmt(s["success_rate"], "%.1f"),
            _fmt(s["length_mean_common"]), _fmt(s["length_std_common"]),
            _fmt(s["length_norm_mean"], "%.3f"),
            _fmt(s["first_time_median"], "%.4f"),
            _fmt(s["time_median"], "%.4f"),
            _fmt(s["nodes_mean"], "%.0f")))

    lines.append("\n### By obstacle density\n")
    lines.append("| density | planner | success | length (own) | length (common) | "
                 "norm. | first path (s) | total (s) | nodes | motion checks |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for density in densities:
        for s in [s for s in summary if s["density_label"] == density]:
            lines.append("| %s | %s | %s%% | %s | %s | %s× | %s | %s | %s | %s |" % (
                density, s["planner"], _fmt(s["success_rate"], "%.0f"),
                _fmt(s["length_mean"]), _fmt(s["length_mean_common"]),
                _fmt(s["length_norm_mean"], "%.3f"),
                _fmt(s["first_time_median"], "%.4f"),
                _fmt(s["time_median"], "%.4f"),
                _fmt(s["nodes_mean"], "%.0f"),
                _fmt(s["motion_checks_mean"], "%.0f")))

    common = common_solved_instances(rows)
    total_instances = len({instance_key(r) for r in rows})
    lines.append("\n_Path length on the common subset is averaged over the %d of %d "
                 "instances that all three planners solved; averaging each planner "
                 "over its own successes alone would flatter whichever planner "
                 "fails on the hardest worlds._" % (len(common), total_instances))

    return "\n".join(lines)


def write_markdown(summary, rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown_tables(summary, rows))
        f.write("\n")
    return path


def print_summary(summary):
    """Console view of the headline table."""
    print("\n%-6s %-10s %8s %12s %12s %10s %10s %9s"
          % ("alg", "density", "success", "len(own)", "len(common)", "norm",
             "first_s", "nodes"))
    print("-" * 88)
    for s in summary:
        print("%-6s %-10s %7.0f%% %12s %12s %10s %10s %9s" % (
            s["planner"], s["density_label"], s["success_rate"],
            _fmt(s["length_mean"]), _fmt(s["length_mean_common"]),
            _fmt(s["length_norm_mean"], "%.3f"),
            _fmt(s["first_time_median"], "%.4f"),
            _fmt(s["nodes_mean"], "%.0f")))
