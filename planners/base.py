"""
The interface all three planners share.

Every planner exposes exactly:

    plan(env, start=None, goal=None) -> (path, metrics)

where `path` is a list of (x, y) waypoints beginning at the start and ending
at the goal, or None on failure, and `metrics` is a flat dict with a fixed
schema so the benchmark harness can write rows straight to CSV without
special-casing any planner.

Keeping the signature identical is what lets the harness treat A*, RRT and
RRT* as interchangeable. Anything planner-specific (grid resolution, step
size, rewiring radius) is a constructor argument, not a plan() argument.
"""

import time

from env.geometry import distance

# Fixed metric schema. Every planner fills every key, so CSV rows line up even
# though the three algorithms count very different internal work.
METRIC_FIELDS = [
    "planner",          # algorithm name
    "environment",      # world name
    "success",          # did it return a valid path within the budget
    "path_length",      # total Euclidean length of the returned path
    "time_sec",         # wall-clock planning time
    "iterations",       # A*: pop count | RRT/RRT*: sample iterations
    "nodes",            # A*: cells expanded | RRT/RRT*: nodes in the tree
    "state_checks",     # point-collision queries made against the Environment
    "motion_checks",    # segment-collision queries made against the Environment
    "reason",           # why it failed (empty string on success)
    # Anytime-planner fields. Only RRT* refines a path after finding one, so
    # for A* and RRT these equal the final values (or NaN on failure). They let
    # the report separate "time to first feasible path" from "time spent
    # improving it", without which RRT* looks unfairly slow next to RRT.
    "first_solution_time",
    "first_solution_length",
    "first_solution_iteration",
]


def path_length(path):
    """Total Euclidean length of a polyline. 0.0 for None/short paths."""
    if not path or len(path) < 2:
        return 0.0
    return sum(distance(path[i], path[i + 1]) for i in range(len(path) - 1))


def make_metrics(**kwargs):
    """Build a metrics dict with every schema field present."""
    row = {
        "planner": "", "environment": "", "success": False, "path_length": float("nan"),
        "time_sec": 0.0, "iterations": 0, "nodes": 0,
        "state_checks": 0, "motion_checks": 0, "reason": "",
        "first_solution_time": float("nan"),
        "first_solution_length": float("nan"),
        "first_solution_iteration": -1,
    }
    unknown = set(kwargs) - set(row)
    if unknown:
        raise KeyError("unknown metric field(s): %s" % ", ".join(sorted(unknown)))
    row.update(kwargs)
    return row


class Planner:
    """Base class holding the bits all three planners genuinely share."""

    name = "planner"

    def __init__(self, time_budget=10.0):
        # Success rate is defined as "found a valid path within a time budget",
        # so the budget belongs to every planner, not just the sampling ones.
        self.time_budget = float(time_budget)

        # Populated by plan() so the visualisation layer can draw internals
        # (the search tree, the set of expanded cells) after a run.
        self.last_env = None

    def plan(self, env, start=None, goal=None):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _start_goal(self, env, start, goal):
        """Resolve the start/goal overrides against the environment defaults."""
        s = tuple(start) if start is not None else env.start
        g = tuple(goal) if goal is not None else env.goal
        return s, g

    def _begin(self, env):
        """Reset per-run bookkeeping and return the start timestamp.

        Collision counters are reset here so each planner is charged only for
        the work of its own run -- otherwise a planner run second on the same
        Environment would inherit the first one's tally.
        """
        env.reset_counters()
        self.last_env = env
        return time.perf_counter()

    def _finish(self, env, t0, path, iterations, nodes, reason="", **extra):
        """Assemble the metrics row shared by all planners.

        `extra` carries planner-specific schema fields (the anytime ones). For
        a non-anytime planner they default to the final result, so every row
        is directly comparable.
        """
        elapsed = time.perf_counter() - t0
        success = path is not None
        if success:
            extra.setdefault("first_solution_time", elapsed)
            extra.setdefault("first_solution_length", path_length(path))
            extra.setdefault("first_solution_iteration", iterations)
        return make_metrics(
            planner=self.name,
            environment=getattr(env, "name", "env"),
            success=success,
            path_length=path_length(path) if success else float("nan"),
            time_sec=elapsed,
            iterations=iterations,
            nodes=nodes,
            state_checks=env.state_checks,
            motion_checks=env.motion_checks,
            reason=reason,
            **extra
        )

    def _out_of_time(self, t0):
        return (time.perf_counter() - t0) > self.time_budget

    def __repr__(self):
        return "%s(time_budget=%g)" % (type(self).__name__, self.time_budget)


def verify_path(env, path, tol=1e-9):
    """Independently re-check that a returned path is actually collision-free.

    Used by the tests and the benchmark harness as a guard against a planner
    reporting success on an invalid path -- which would silently corrupt both
    the success-rate and the path-length comparison.

    Returns (ok, reason).
    """
    if path is None:
        return False, "path is None"
    if len(path) < 2:
        return False, "path has fewer than 2 waypoints"

    if not env.is_state_valid(path[0]):
        return False, "first waypoint is in collision"
    if not env.is_state_valid(path[-1]):
        return False, "last waypoint is in collision"

    for i in range(len(path) - 1):
        if not env.is_motion_valid(path[i], path[i + 1]):
            return False, "segment %d -> %d is in collision" % (i, i + 1)

    if distance(path[0], env.start) > tol:
        return False, "path does not begin at the start"
    if distance(path[-1], env.goal) > tol:
        return False, "path does not end at the goal"

    return True, ""
