"""
Benchmark harness: run every planner over an identical set of worlds.

The design rule is that nothing about a run may differ between planners except
the algorithm itself. Concretely:

  * The same `generate_suite` world list goes to all three planners.
  * Trial `t` uses seed `t` for every stochastic planner, so RRT and RRT* see
    the same random stream on the same world.
  * Each planner gets the same time budget and the same iteration cap.
  * Every returned path is re-verified before its metrics are recorded.

Output is a tidy CSV -- one row per (planner, world, trial) -- which the
analysis stage aggregates. Raw rows are kept rather than summaries so the
numbers in the report can always be traced back to individual runs.
"""

import csv
import json
import os
import time

from env import generate_suite
from planners import METRIC_FIELDS, AStarPlanner, RRTPlanner, RRTStarPlanner, verify_path

# Columns describing *which run this was*, prepended to the planner's own metrics.
CONTEXT_FIELDS = ["density_label", "density_target", "trial", "seed"]
CSV_FIELDS = CONTEXT_FIELDS + METRIC_FIELDS


class BenchmarkConfig:
    """All knobs in one place, so a run is fully described by this object.

    Defaults are the project's headline settings. `to_dict` is written
    alongside the results so any table in the report can be reproduced.
    """

    def __init__(self, densities=None, n_envs_per_density=10, n_trials=5,
                 time_budget=10.0, max_iters=10000, astar_resolution=1.0,
                 step_size=5.0, goal_bias=0.05, goal_tolerance=5.0,
                 suite_seed=0, out_dir="results"):
        self.densities = densities or {"sparse": 0.10, "moderate": 0.20, "cluttered": 0.30}
        self.n_envs_per_density = n_envs_per_density
        self.n_trials = n_trials
        self.time_budget = time_budget
        self.max_iters = max_iters
        self.astar_resolution = astar_resolution
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance
        self.suite_seed = suite_seed
        self.out_dir = out_dir

    def make_planners(self, seed):
        """The three planners for one trial, sharing every common parameter.

        Sharing step_size / goal_bias / goal_tolerance / max_iters between RRT
        and RRT* is what makes their comparison about rewiring rather than
        about tuning.
        """
        return [
            AStarPlanner(resolution=self.astar_resolution, connectivity=8,
                         heuristic="octile", time_budget=self.time_budget),
            RRTPlanner(step_size=self.step_size, goal_bias=self.goal_bias,
                       goal_tolerance=self.goal_tolerance, max_iters=self.max_iters,
                       time_budget=self.time_budget, seed=seed),
            RRTStarPlanner(step_size=self.step_size, goal_bias=self.goal_bias,
                           goal_tolerance=self.goal_tolerance, max_iters=self.max_iters,
                           time_budget=self.time_budget, seed=seed),
        ]

    def to_dict(self):
        return dict(densities=self.densities,
                    n_envs_per_density=self.n_envs_per_density,
                    n_trials=self.n_trials, time_budget=self.time_budget,
                    max_iters=self.max_iters, astar_resolution=self.astar_resolution,
                    step_size=self.step_size, goal_bias=self.goal_bias,
                    goal_tolerance=self.goal_tolerance, suite_seed=self.suite_seed)


def run_benchmark(config=None, verbose=True):
    """Run the full sweep. Returns (rows, convergence, suite).

    `convergence` holds RRT*'s best-cost-over-time curves, which are what
    demonstrate the anytime behaviour that a single final number hides.
    """
    config = config or BenchmarkConfig()

    if verbose:
        print("Generating %d worlds (%d per density x %d densities)..."
              % (config.n_envs_per_density * len(config.densities),
                 config.n_envs_per_density, len(config.densities)))

    suite = generate_suite(densities=config.densities,
                           n_per_density=config.n_envs_per_density,
                           seed=config.suite_seed)

    total = len(suite) * config.n_trials * 3
    rows = []
    convergence = []
    done = 0
    t_start = time.perf_counter()

    for env in suite:
        for trial in range(config.n_trials):
            for planner in config.make_planners(seed=trial):
                path, metrics = planner.plan(env)

                # A planner claiming success on an invalid path would corrupt
                # both the success-rate and the path-length columns, so the
                # claim is checked rather than trusted. verify_path runs after
                # the metrics dict is built, so its own collision queries do
                # not pollute the recorded counts.
                if metrics["success"]:
                    ok, why = verify_path(env, path)
                    if not ok:
                        raise AssertionError(
                            "%s returned an invalid path on %s (trial %d): %s"
                            % (metrics["planner"], env.name, trial, why))

                row = {
                    "density_label": getattr(env, "density_label", ""),
                    "density_target": getattr(env, "density_target", float("nan")),
                    "trial": trial,
                    "seed": trial,
                }
                row.update(metrics)
                rows.append(row)

                # Keep RRT*'s refinement curve for the convergence figure.
                if metrics["planner"] == "RRT*" and planner.cost_history:
                    convergence.append({
                        "environment": env.name,
                        "density_label": row["density_label"],
                        "trial": trial,
                        "history": [[int(i), float(t), float(c)]
                                    for i, t, c in planner.cost_history],
                    })

                done += 1
                if verbose and done % 25 == 0:
                    elapsed = time.perf_counter() - t_start
                    rate = done / elapsed
                    print("  %4d/%4d runs  (%.1f%%)  %.1f s elapsed, ~%.0f s left"
                          % (done, total, 100.0 * done / total, elapsed,
                             (total - done) / rate if rate else 0))

    if verbose:
        print("Completed %d runs in %.1f s" % (total, time.perf_counter() - t_start))
    return rows, convergence, suite


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------

def write_rows(rows, path):
    """Write the raw per-run results as CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_rows(path):
    """Read raw results back, restoring numeric types."""
    ints = {"trial", "seed", "iterations", "nodes", "state_checks",
            "motion_checks", "first_solution_iteration"}
    floats = {"density_target", "path_length", "time_sec",
              "first_solution_time", "first_solution_length"}

    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            parsed["success"] = row["success"] == "True"
            for k in ints:
                parsed[k] = int(row[k])
            for k in floats:
                parsed[k] = float(row[k])
            out.append(parsed)
    return out


def write_run_metadata(config, suite, path):
    """Record the configuration and the exact worlds used.

    Without this a results table is not reproducible: the numbers depend on
    which worlds were drawn, and those come from the suite seed.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "environments": [
            {"name": e.name,
             "density_label": getattr(e, "density_label", ""),
             "density_target": getattr(e, "density_target", None),
             "density_actual": e.obstacle_density(),
             "n_obstacles": len(e.obstacles),
             "start": list(e.start), "goal": list(e.goal)}
            for e in suite
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def write_convergence(convergence, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(convergence, f)
    return path


def read_convergence(path):
    with open(path) as f:
        return json.load(f)
