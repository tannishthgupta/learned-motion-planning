"""
Tests for Informed RRT* (planners.samplers.InformedSampler).

What these protect:

  * Before the first solution it behaves exactly like plain RRT* -- same
    samples, same first path. Informed RRT* only differs *after* that.
  * After a solution of cost c, every sample lies inside the ellipse
    |x - start| + |x - goal| <= c, and inside the workspace.
  * The ellipse is sampled uniformly (not bunched at the centre), which is
    what the paper's direct-sampling method guarantees.
  * Wrapping another sampler (e.g. the learned heatmap) keeps its samples
    inside the ellipse too.
  * Paths are still valid.
"""

import math
import random
import unittest

import numpy as np

from env import Environment, get_scenario
from planners import (HeatmapSampler, InformedSampler, RRTStarPlanner,
                      UniformSampler, verify_path)


def _env():
    return Environment(100, 100, [], (10, 20), (80, 60), robot_radius=1.0)


class TestInformedSampler(unittest.TestCase):

    def test_identical_to_base_before_first_solution(self):
        env = _env()
        s = InformedSampler()
        s.prepare(env, env.start, env.goal)
        r1, r2 = random.Random(7), random.Random(7)
        for _ in range(200):
            self.assertEqual(s.sample(env, r1), UniformSampler().sample(env, r2))

    def test_first_solution_unchanged_in_rrt_star(self):
        env = get_scenario("cluttered_discs")
        _, a = RRTStarPlanner(seed=4, max_iters=3000).plan(env)
        _, b = RRTStarPlanner(seed=4, max_iters=3000, sampler=InformedSampler()).plan(env)
        self.assertEqual(a["first_solution_iteration"], b["first_solution_iteration"])
        self.assertAlmostEqual(a["first_solution_length"], b["first_solution_length"])

    def test_samples_stay_inside_ellipse_and_bounds(self):
        env = _env()
        s = InformedSampler()
        s.prepare(env, env.start, env.goal)
        c_min = math.dist(env.start, env.goal)
        s.notify_best_cost(1.2 * c_min)
        rng = random.Random(0)
        for _ in range(2000):
            p = s.sample(env, rng)
            self.assertLessEqual(math.dist(p, env.start) + math.dist(p, env.goal),
                                 1.2 * c_min + 1e-9)
            self.assertTrue(env.in_bounds(p))

    def test_ellipse_shrinks_and_never_grows(self):
        s = InformedSampler()
        s.prepare(_env(), (10, 20), (80, 60))
        s.notify_best_cost(150)
        s.notify_best_cost(120)
        s.notify_best_cost(140)          # a worse cost must not widen it
        self.assertEqual(s.c_best, 120)

    def test_uniform_inside_ellipse(self):
        # For a uniform distribution, the inner ellipse scaled by 1/2 holds
        # 1/4 of the area, so ~25% of samples should fall in it. Bunching at
        # the centre (forgetting the sqrt) would give ~50%.
        env = Environment(1000, 1000, [], (400, 500), (600, 500))
        s = InformedSampler()
        s.prepare(env, env.start, env.goal)
        s.notify_best_cost(300.0)                    # r1 = 150, r2 = ~111.8
        r1, r2 = 150.0, math.sqrt(300.0 ** 2 - 200.0 ** 2) / 2
        rng = random.Random(1)
        n, inner = 20000, 0
        for _ in range(n):
            x, y = s.sample(env, rng)
            u, v = (x - 500) / r1, (y - 500) / r2
            inner += (u * u + v * v) <= 0.25
        self.assertAlmostEqual(inner / n, 0.25, delta=0.015)

    def test_wrapped_heatmap_samples_stay_inside(self):
        env = _env()
        heat = np.ones((10, 10))
        s = InformedSampler(base=HeatmapSampler(heat, mix=0.9))
        s.prepare(env, env.start, env.goal)
        s.notify_best_cost(1.1 * math.dist(env.start, env.goal))
        rng = random.Random(2)
        for _ in range(1000):
            self.assertTrue(s.in_ellipse(s.sample(env, rng)))


class TestInformedRRTStar(unittest.TestCase):

    def test_returns_valid_path_and_monotone_cost(self):
        env = get_scenario("bug_trap")
        p = RRTStarPlanner(seed=0, max_iters=4000, sampler=InformedSampler())
        path, m = p.plan(env)
        self.assertTrue(m["success"])
        self.assertTrue(verify_path(env, path)[0])
        costs = [c for _, _, c in p.cost_history]
        self.assertEqual(costs, sorted(costs, reverse=True))
        # The sampler saw the final cost.
        self.assertAlmostEqual(p.sampler.c_best, costs[-1])

    def test_improves_on_uniform_in_open_space(self):
        # In an empty world the ellipse is the whole story, so Informed RRT*
        # should beat uniform RRT* at the same iteration budget on average.
        env = get_scenario("empty")
        u, i = [], []
        for seed in range(6):
            u.append(RRTStarPlanner(seed=seed, max_iters=1500).plan(env)[1]["path_length"])
            i.append(RRTStarPlanner(seed=seed, max_iters=1500,
                                    sampler=InformedSampler()).plan(env)[1]["path_length"])
        self.assertLess(np.mean(i), np.mean(u))


if __name__ == "__main__":
    unittest.main()
