"""
End-to-end demo: build worlds, run all three planners, render the figures.

This is the "does it all work" entry point, not the benchmark. It runs each
planner once per scenario and writes comparison figures to results/.

Usage:
    python demo.py                      # all fixed scenarios
    python demo.py --scenario maze      # one scenario
    python demo.py --random 0.25        # a random world at 25% obstacle density
    python demo.py --budget 5           # per-planner time budget in seconds
"""

import argparse
import os

from env import SCENARIOS, get_scenario, solvable_random_environment
from planners import default_planners, verify_path
from viz import compare_planners_figure, overlay_paths_figure, save

RESULTS_DIR = "results"


def run_all(env, seed=42, budget=10.0):
    """Run every planner on one world; returns a list of (planner, path, metrics).

    Each planner gets a freshly constructed Environment-level counter reset
    (handled inside Planner._begin), so the collision-check numbers below
    describe that planner alone.
    """
    runs = []
    for planner in default_planners(seed=seed, time_budget=budget):
        path, metrics = planner.plan(env)

        # Independent re-verification. A planner claiming success on a path
        # that clips an obstacle would quietly invalidate every metric, so the
        # check is not optional.
        if metrics["success"]:
            ok, why = verify_path(env, path)
            if not ok:
                raise AssertionError("%s returned an invalid path on %s: %s"
                                     % (metrics["planner"], env.name, why))
        runs.append((planner, path, metrics))
    return runs


def print_table(env, runs):
    """Print the per-planner metrics for one world."""
    print("\n%s  (%d obstacles, density %.2f)"
          % (env.name, len(env.obstacles), env.obstacle_density()))
    print("  %-5s %9s %9s %9s %8s %10s %10s"
          % ("alg", "length", "time_s", "1st_len", "nodes", "state_chk", "motion_chk"))
    print("  " + "-" * 68)
    for _, _, m in runs:
        if m["success"]:
            first = ("%.2f" % m["first_solution_length"])
            print("  %-5s %9.2f %9.3f %9s %8d %10d %10d"
                  % (m["planner"], m["path_length"], m["time_sec"], first,
                     m["nodes"], m["state_checks"], m["motion_checks"]))
        else:
            print("  %-5s %9s %9.3f %9s %8d %10d %10d   (%s)"
                  % (m["planner"], "FAILED", m["time_sec"], "-", m["nodes"],
                     m["state_checks"], m["motion_checks"], m["reason"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS),
                    help="run a single fixed scenario")
    ap.add_argument("--random", type=float, metavar="DENSITY",
                    help="run a random world at this obstacle density")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=float, default=10.0,
                    help="per-planner time budget in seconds")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if args.random is not None:
        envs = [solvable_random_environment(density=args.random, seed=args.seed)[0]]
    elif args.scenario:
        envs = [get_scenario(args.scenario)]
    else:
        envs = [get_scenario(n) for n in
                ["empty", "cluttered_discs", "maze", "narrow_gap", "bug_trap"]]

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for env in envs:
        runs = run_all(env, seed=args.seed, budget=args.budget)
        print_table(env, runs)

        if args.no_figures:
            continue

        fig = compare_planners_figure(env, runs)
        out = save(fig, os.path.join(RESULTS_DIR, "%s_panels.png" % env.name))
        print("  -> %s" % out)

        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(6.5, 6.5))
        overlay_paths_figure(env, runs, ax=ax)
        out = save(ax.figure, os.path.join(RESULTS_DIR, "%s_overlay.png" % env.name))
        print("  -> %s" % out)

    print("\nAll paths independently verified collision-free.")


if __name__ == "__main__":
    main()
