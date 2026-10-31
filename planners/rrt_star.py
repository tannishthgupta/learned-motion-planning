"""
RRT* -- asymptotically optimal Rapidly-exploring Random Tree.

Reference: Karaman & Frazzoli, "Sampling-Based Algorithms for Optimal Motion
Planning", International Journal of Robotics Research 30(7), 2011.

RRT* keeps RRT's sample/nearest/steer skeleton and adds two steps that turn a
feasible planner into an asymptotically optimal one:

    CHOOSE PARENT  Instead of attaching the new node to its nearest neighbour,
                   consider every node within a shrinking ball and attach to
                   whichever yields the lowest cost-to-come.

    REWIRE         Then check the reverse: for each node in that same ball,
                   would routing it *through* the new node be cheaper? If so,
                   re-attach it and propagate the saving to its descendants.

Rewiring is what makes the difference. RRT's tree is frozen once built -- an
early bad edge is permanent. RRT* continuously repairs its own topology, so
the cost of the best path converges to the optimum as the number of samples
goes to infinity.

The shrinking ball radius is the theoretical heart of the algorithm:

    r(n) = min( gamma * (log(n) / n)^(1/d),  eta )

It must shrink (or the per-iteration cost grows without bound) but shrink
slowly enough that the connectivity needed for optimality is retained. Karaman
& Frazzoli show the constant must satisfy

    gamma > gamma* = 2 * (1 + 1/d)^(1/d) * (mu(X_free) / zeta_d)^(1/d)

with d the dimension, mu(X_free) the free-space volume and zeta_d the volume
of the unit d-ball. For our planar case d = 2 and zeta_2 = pi, so this is
computed directly from the environment rather than hand-tuned.

Cost model: path length. `cost[i]` is the length of the tree path from the
root to node i, so minimising cost minimises distance travelled.
"""

import math
import random
import time

from env.geometry import distance

from .base import Planner
from .rrt import steer
from .samplers import UniformSampler
from .tree import Tree


def gamma_star(free_area, dimension=2):
    """The theoretical lower bound on the rewiring constant (see module docs).

    For d = 2:  gamma* = 2 * sqrt(1.5) * sqrt(free_area / pi)
    """
    d = dimension
    unit_ball_volume = math.pi if d == 2 else math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1)
    return 2.0 * (1.0 + 1.0 / d) ** (1.0 / d) * (free_area / unit_ball_volume) ** (1.0 / d)


class RRTStarPlanner(Planner):
    """RRT* with choose-parent, rewiring, and anytime refinement.

    Parameters
    ----------
    step_size : float
        Steering range eta. Also caps the rewiring radius, per the theory:
        connecting further than the planner can steer would be inconsistent.
    goal_bias, goal_tolerance, max_iters, time_budget, seed
        As in RRTPlanner, so the two are directly comparable.
    gamma_factor : float
        Safety multiplier on gamma*. Must be > 1 for the optimality guarantee;
        larger values rewire over a wider ball (better paths, slower).
    stop_on_first_solution : bool
        If True, return as soon as any path is found, exactly like RRT. This
        is off by default because it discards the algorithm's whole point --
        but it is available for a like-for-like time comparison against RRT,
        where letting RRT* spend the full budget refining would be misleading.
    sampler : object or None
        Strategy for the SAMPLE step (see planners/samplers.py). None means
        uniform sampling, as in Karaman & Frazzoli.
    name : str or None
        Label used in metrics rows (e.g. "RRT* (learned)").
    """

    name = "RRT*"

    def __init__(self, step_size=5.0, goal_bias=0.05, goal_tolerance=5.0,
                 max_iters=10000, time_budget=10.0, seed=None,
                 gamma_factor=1.5, stop_on_first_solution=False,
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
        self.gamma_factor = float(gamma_factor)
        self.stop_on_first_solution = bool(stop_on_first_solution)

        # Exposed for visualisation / convergence analysis after a run.
        self.tree = None
        self.cost_history = []   # (iteration, elapsed_sec, best_cost)

    # ------------------------------------------------------------------

    def _compute_gamma(self, env):
        """Derive the rewiring constant from this environment's free area."""
        free_area = env.width * env.height * max(0.05, 1.0 - env.obstacle_density())
        return self.gamma_factor * gamma_star(free_area, dimension=2)

    def _radius(self, gamma, n):
        """Shrinking ball radius r(n), capped at the steering range."""
        if n <= 1:
            return self.step_size
        return min(gamma * math.sqrt(math.log(n) / n), self.step_size)

    # ------------------------------------------------------------------

    def plan(self, env, start=None, goal=None):
        start_pt, goal_pt = self._start_goal(env, start, goal)
        t0 = self._begin(env)

        rng = random.Random(self.seed)
        self.tree = Tree(start_pt)
        self.cost_history = []
        tree = self.tree

        gamma = self._compute_gamma(env)

        # Sampler set-up runs inside the timer (see RRTPlanner.plan).
        self.sampler.prepare(env, start_pt, goal_pt)

        goal_idx = None              # index of the goal once it joins the tree
        first_time = float("nan")
        first_length = float("nan")
        first_iter = -1

        iterations = 0

        for iterations in range(1, self.max_iters + 1):
            if iterations % 64 == 0 and self._out_of_time(t0):
                break

            # --- SAMPLE / NEAREST / STEER (identical to RRT) ---------------
            if rng.random() < self.goal_bias:
                sample = goal_pt
            else:
                sample = self.sampler.sample(env, rng)

            nearest_idx = tree.nearest(sample)
            new_pt = steer(tree.point(nearest_idx), sample, self.step_size)

            if not env.is_state_valid(new_pt):
                continue
            if not env.is_motion_valid(tree.point(nearest_idx), new_pt):
                continue

            # --- Neighbourhood for rewiring -------------------------------
            radius = self._radius(gamma, len(tree))
            near_indices = tree.near(new_pt, radius)

            # --- CHOOSE PARENT --------------------------------------------
            # Start from the nearest node (already known collision-free) and
            # look for any neighbour offering a cheaper route to new_pt.
            best_parent = nearest_idx
            best_cost = tree.cost[nearest_idx] + distance(tree.point(nearest_idx), new_pt)

            for idx in near_indices:
                if idx == nearest_idx or idx == goal_idx:
                    continue
                candidate_cost = tree.cost[idx] + distance(tree.point(idx), new_pt)
                # Compare cost before paying for the collision check -- the
                # geometry is cheap, the segment test is not.
                if candidate_cost < best_cost and env.is_motion_valid(tree.point(idx), new_pt):
                    best_parent = idx
                    best_cost = candidate_cost

            new_idx = tree.add_node(new_pt, parent=best_parent, cost=best_cost)

            # --- REWIRE ---------------------------------------------------
            # The reverse question: is going through the new node cheaper for
            # any neighbour? Re-attaching propagates the saving to that node's
            # whole subtree (handled inside Tree.set_parent).
            for idx in near_indices:
                if idx == new_idx or idx == best_parent:
                    continue
                rewired_cost = best_cost + distance(new_pt, tree.point(idx))
                if rewired_cost < tree.cost[idx] and env.is_motion_valid(new_pt, tree.point(idx)):
                    tree.set_parent(idx, new_idx, rewired_cost)

            # --- Goal handling --------------------------------------------
            # The goal lives in the tree as an ordinary node, so rewiring can
            # improve the route to it for free. It is excluded from being a
            # parent, since nothing should route *through* the goal.
            if distance(new_pt, goal_pt) <= self.goal_tolerance:
                cost_via_new = tree.cost[new_idx] + distance(new_pt, goal_pt)

                if goal_idx is None:
                    if env.is_motion_valid(new_pt, goal_pt):
                        goal_idx = tree.add_node(goal_pt, parent=new_idx, cost=cost_via_new)
                        first_time = time.perf_counter() - t0
                        first_length = tree.cost[goal_idx]
                        first_iter = iterations
                        self.cost_history.append((iterations, first_time, first_length))
                        self.sampler.notify_best_cost(first_length)
                        if self.stop_on_first_solution:
                            break
                elif cost_via_new < tree.cost[goal_idx]:
                    if env.is_motion_valid(new_pt, goal_pt):
                        tree.set_parent(goal_idx, new_idx, cost_via_new)

            # Record the best-cost curve whenever it improves. This is the
            # convergence evidence for the report: RRT* should show a
            # monotonically decreasing cost, RRT a flat line at its first hit.
            if goal_idx is not None:
                best = tree.cost[goal_idx]
                if not self.cost_history or best < self.cost_history[-1][2] - 1e-12:
                    self.cost_history.append((iterations, time.perf_counter() - t0, best))
                    # Informed RRT* shrinks its sampling ellipse here.
                    self.sampler.notify_best_cost(best)

        if goal_idx is None:
            reason = ("time budget exceeded" if self._out_of_time(t0)
                      else "iteration limit reached")
            return None, self._finish(env, t0, None, iterations, len(tree), reason)

        path = tree.path_to_root(goal_idx)
        return path, self._finish(
            env, t0, path, iterations, len(tree),
            first_solution_time=first_time,
            first_solution_length=first_length,
            first_solution_iteration=first_iter,
        )

    def __repr__(self):
        return ("RRTStarPlanner(step_size=%g, goal_bias=%g, goal_tolerance=%g, "
                "max_iters=%d, gamma_factor=%g)"
                % (self.step_size, self.goal_bias, self.goal_tolerance,
                   self.max_iters, self.gamma_factor))
