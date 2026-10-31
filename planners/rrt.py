"""
Rapidly-exploring Random Tree (RRT).

Reference: LaValle, "Rapidly-Exploring Random Trees: A New Tool for Path
Planning", TR 98-11, Iowa State University, 1998.

The loop is four steps:

    1. SAMPLE  a random configuration from the workspace
    2. NEAREST find the tree node closest to it
    3. STEER   move a bounded step from that node toward the sample
    4. EXTEND  keep the new node if both it and the edge are collision-free

The rapid-exploration property comes from step 2: the probability of a node
being selected is proportional to the area of its Voronoi region, so the tree
is pulled toward large unexplored regions without any explicit frontier
bookkeeping.

RRT is *probabilistically complete* -- given enough samples it finds a path if
one exists -- but it is **not** optimal. It returns the first path it stumbles
into, which is typically jagged and noticeably longer than necessary. That
non-optimality is precisely what RRT* fixes, and it is the main axis of this
project's comparison.

Implementation note: this planner stops at the first path found. That is the
textbook formulation, and keeping it unmodified is what makes the RRT vs RRT*
path-length gap a fair measurement rather than an artefact of tuning.
"""

import math
import random

from env.geometry import distance

from .base import Planner
from .samplers import UniformSampler
from .tree import Tree


def steer(from_p, to_p, step_size):
    """Return a point at most `step_size` from `from_p`, heading toward `to_p`.

    If the target is already within one step, it is returned unchanged -- this
    lets the tree land exactly on the goal rather than perpetually overshooting
    it by a fraction of a step.
    """
    dx = to_p[0] - from_p[0]
    dy = to_p[1] - from_p[1]
    d = math.hypot(dx, dy)

    if d <= step_size or d == 0.0:
        return (to_p[0], to_p[1])

    scale = step_size / d
    return (from_p[0] + dx * scale, from_p[1] + dy * scale)


class RRTPlanner(Planner):
    """Classic single-tree RRT with goal biasing.

    Parameters
    ----------
    step_size : float
        Maximum extension length per iteration. Small values explore finely
        but need many iterations; large values may fail to thread gaps.
    goal_bias : float
        Probability of sampling the goal itself instead of a random point.
        A small bias speeds convergence markedly in open space; too large a
        value degenerates into greedy search and gets stuck in concave traps
        (see the `bug_trap` scenario).
    goal_tolerance : float
        A node this close to the goal may attempt a direct connection to it.
    max_iters : int
        Iteration cap. Together with `time_budget` this defines the failure
        condition used by the success-rate metric.
    seed : int or None
        Seeds this planner's own RNG. Each run gets an independent stream, so
        trials are reproducible and the RNG state never leaks between runs.
    sampler : object or None
        Strategy for step 1 (see planners/samplers.py). None means uniform
        sampling, i.e. the textbook algorithm.
    name : str or None
        Label used in metrics rows, so variants with different samplers can
        be told apart in the benchmark output.
    """

    name = "RRT"

    def __init__(self, step_size=5.0, goal_bias=0.05, goal_tolerance=5.0,
                 max_iters=10000, time_budget=10.0, seed=None,
                 sampler=None, name=None):
        super().__init__(time_budget=time_budget)
        self.sampler = sampler if sampler is not None else UniformSampler()
        if name is not None:
            self.name = name
        self.step_size = float(step_size)
        self.goal_bias = float(goal_bias)
        self.goal_tolerance = float(goal_tolerance)
        self.max_iters = int(max_iters)
        self.seed = seed

        # Exposed for visualisation after a run.
        self.tree = None

    # ------------------------------------------------------------------

    def plan(self, env, start=None, goal=None):
        start_pt, goal_pt = self._start_goal(env, start, goal)
        t0 = self._begin(env)

        rng = random.Random(self.seed)
        self.tree = Tree(start_pt)
        tree = self.tree

        # Sampler set-up runs inside the timer, so a learned sampler's
        # inference cost counts toward this planner's computation time.
        self.sampler.prepare(env, start_pt, goal_pt)

        iterations = 0

        for iterations in range(1, self.max_iters + 1):
            if iterations % 64 == 0 and self._out_of_time(t0):
                return None, self._finish(env, t0, None, iterations, len(tree),
                                          "time budget exceeded")

            # --- 1. SAMPLE ------------------------------------------------
            # Goal biasing: occasionally aim straight at the goal. The rest of
            # the time sample uniformly, which is what preserves coverage of
            # the free space and hence probabilistic completeness.
            if rng.random() < self.goal_bias:
                sample = goal_pt
            else:
                sample = self.sampler.sample(env, rng)

            # --- 2. NEAREST -----------------------------------------------
            nearest_idx = tree.nearest(sample)
            nearest_pt = tree.point(nearest_idx)

            # --- 3. STEER -------------------------------------------------
            new_pt = steer(nearest_pt, sample, self.step_size)

            # --- 4. EXTEND ------------------------------------------------
            # Both the new configuration and the swept edge must be free.
            # Checking the state first is the cheaper test, so it short-
            # circuits the more expensive segment check.
            if not env.is_state_valid(new_pt):
                continue
            if not env.is_motion_valid(nearest_pt, new_pt):
                continue

            new_idx = tree.add_node(
                new_pt, parent=nearest_idx,
                cost=tree.cost[nearest_idx] + distance(nearest_pt, new_pt))

            # --- Goal check -----------------------------------------------
            # Try to close the gap directly. RRT terminates on the first
            # success: no attempt is made to improve the path afterwards.
            if distance(new_pt, goal_pt) <= self.goal_tolerance:
                if env.is_motion_valid(new_pt, goal_pt):
                    goal_idx = tree.add_node(
                        goal_pt, parent=new_idx,
                        cost=tree.cost[new_idx] + distance(new_pt, goal_pt))
                    path = tree.path_to_root(goal_idx)
                    return path, self._finish(env, t0, path, iterations, len(tree))

        return None, self._finish(env, t0, None, iterations, len(tree),
                                  "iteration limit reached")

    def __repr__(self):
        return ("RRTPlanner(step_size=%g, goal_bias=%g, goal_tolerance=%g, "
                "max_iters=%d)" % (self.step_size, self.goal_bias,
                                   self.goal_tolerance, self.max_iters))
