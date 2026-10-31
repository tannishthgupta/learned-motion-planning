"""
Motion planners, all sharing the `plan(env, start, goal) -> (path, metrics)`
interface defined in `planners.base`.

    A*    -- resolution-optimal graph search on a grid       (astar.py)
    RRT   -- probabilistically complete sampling planner     (rrt.py)
    RRT*  -- asymptotically optimal sampling planner         (rrt_star.py)
"""

from .astar import AStarPlanner
from .base import METRIC_FIELDS, Planner, make_metrics, path_length, verify_path
from .rrt import RRTPlanner
from .rrt_star import RRTStarPlanner
from .samplers import HeatmapSampler, UniformSampler
from .tree import Tree

#: Canonical ordering used by the benchmark and every plot legend.
PLANNER_ORDER = ["A*", "RRT", "RRT*"]


def default_planners(seed=None, time_budget=10.0):
    """The three planners with the project's default settings.

    Shared parameters are deliberately identical across the sampling planners
    (step size, goal bias, goal tolerance, iteration cap) so that any measured
    difference between RRT and RRT* comes from the rewiring, not from tuning.
    """
    return [
        AStarPlanner(resolution=1.0, connectivity=8, time_budget=time_budget),
        RRTPlanner(step_size=5.0, goal_bias=0.05, goal_tolerance=5.0,
                   max_iters=10000, time_budget=time_budget, seed=seed),
        RRTStarPlanner(step_size=5.0, goal_bias=0.05, goal_tolerance=5.0,
                       max_iters=10000, time_budget=time_budget, seed=seed),
    ]


__all__ = [
    "AStarPlanner", "RRTPlanner", "RRTStarPlanner",
    "UniformSampler", "HeatmapSampler",
    "Planner", "Tree", "path_length", "verify_path", "make_metrics",
    "METRIC_FIELDS", "PLANNER_ORDER", "default_planners",
]
