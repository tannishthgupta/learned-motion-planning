"""
Run the full benchmark and produce every table and figure the report needs.

Usage:
    python run_benchmark.py                  # full sweep, then tables + charts
    python run_benchmark.py --quick          # small sweep, for a sanity check
    python run_benchmark.py --from-cache     # re-analyse the last saved run
    python run_benchmark.py --envs 20 --trials 10

`--from-cache` matters in practice: the sweep takes minutes, and re-running it
just to restyle a chart would produce *different numbers* (different worlds,
different seeds), silently invalidating any figure already pasted into the
report. Analysis and plotting always work from the saved raw CSV.

Outputs, all in results/:
    benchmark_raw.csv     one row per (planner, world, trial)
    benchmark_meta.json   the config and the exact worlds used
    convergence.json      RRT* best-cost curves
    summary.csv           aggregated metrics
    summary.md            the report tables (also the charts' table view)
    chart_*.png           the figures
"""

import argparse
import os

from benchmark import (BenchmarkConfig, print_summary, read_convergence, read_rows,
                       run_benchmark, summarise, write_convergence, write_markdown,
                       write_rows, write_run_metadata, write_summary_csv)
from viz.charts import make_all_charts

RESULTS = "results"


def _paths(quick=False):
    """Output paths for a run.

    A --quick run writes to its own `*_quick.*` files. Without this it would
    overwrite the headline dataset with a handful of throwaway rows, and the
    next --from-cache would quietly rebuild every table and figure from them --
    invalidating numbers already quoted in the report with no visible error.
    """
    suffix = "_quick" if quick else ""
    return {
        "raw": os.path.join(RESULTS, "benchmark_raw%s.csv" % suffix),
        "meta": os.path.join(RESULTS, "benchmark_meta%s.json" % suffix),
        "conv": os.path.join(RESULTS, "convergence%s.json" % suffix),
        "summary_csv": os.path.join(RESULTS, "summary%s.csv" % suffix),
        "summary_md": os.path.join(RESULTS, "summary%s.md" % suffix),
        "chart_prefix": suffix,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-cache", action="store_true",
                    help="skip the sweep; re-analyse results/benchmark_raw.csv")
    ap.add_argument("--quick", action="store_true",
                    help="small sweep (3 worlds x 2 trials) for a sanity check; "
                         "writes to separate *_quick files so it cannot "
                         "overwrite the headline results")
    ap.add_argument("--envs", type=int, default=10, help="worlds per density level")
    ap.add_argument("--trials", type=int, default=5, help="trials per world")
    ap.add_argument("--budget", type=float, default=10.0, help="time budget, seconds")
    ap.add_argument("--max-iters", type=int, default=10000)
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    paths = _paths(quick=args.quick)

    if args.from_cache:
        if not os.path.exists(paths["raw"]):
            raise SystemExit("no cached results at %s -- run without --from-cache first"
                             % paths["raw"])
        print("Re-analysing cached results from %s" % paths["raw"])
        rows = read_rows(paths["raw"])
        convergence = (read_convergence(paths["conv"])
                       if os.path.exists(paths["conv"]) else [])
    else:
        n_envs = 3 if args.quick else args.envs
        n_trials = 2 if args.quick else args.trials
        config = BenchmarkConfig(n_envs_per_density=n_envs, n_trials=n_trials,
                                 time_budget=args.budget, max_iters=args.max_iters)
        rows, convergence, suite = run_benchmark(config)
        write_rows(rows, paths["raw"])
        write_run_metadata(config, suite, paths["meta"])
        write_convergence(convergence, paths["conv"])
        print("Wrote %s, %s, %s" % (paths["raw"], paths["meta"], paths["conv"]))

    summary = summarise(rows)
    print_summary(summary)

    write_summary_csv(summary, paths["summary_csv"])
    write_markdown(summary, rows, paths["summary_md"])
    print("\nWrote %s and %s" % (paths["summary_csv"], paths["summary_md"]))

    if not args.no_charts:
        for path in make_all_charts(summary, rows, convergence, out_dir=RESULTS,
                                    suffix=paths["chart_prefix"]):
            print("  -> %s" % path)


if __name__ == "__main__":
    main()
