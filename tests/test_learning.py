"""
Tests for the learned-sampling extension.

What these protect:

  * Adding the sampler hook did not change the original planners: with the
    default (uniform) sampler, RRT and RRT* produce exactly the same path as
    before for the same seed.
  * A biased sampler can never produce an invalid path or break the
    guarantees: paths are still verified, and an all-zero heatmap falls back
    to uniform sampling.
  * The training labels mean what the report says they mean: the optimal
    route is always inside the labelled region, and in an empty world the
    region is a band around the straight line.
  * The leakage guard: the train / val / test splits never share a world.

Torch-dependent tests are skipped automatically if PyTorch is not installed,
so the core project test suite still runs on the minimal NumPy + Matplotlib
stack.
"""

import math
import random
import unittest

import numpy as np

from env import Environment, RectObstacle, get_scenario
from planners import (HeatmapSampler, RRTPlanner, RRTStarPlanner, UniformSampler,
                      verify_path)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


class TestSamplerHook(unittest.TestCase):

    def test_default_sampler_is_unchanged_rrt_star(self):
        env = get_scenario("cluttered_discs")
        a, _ = RRTStarPlanner(seed=3, max_iters=800).plan(env)
        b, _ = RRTStarPlanner(seed=3, max_iters=800, sampler=UniformSampler()).plan(env)
        self.assertEqual(a, b)

    def test_default_sampler_is_unchanged_rrt(self):
        env = get_scenario("bug_trap")
        a, _ = RRTPlanner(seed=5).plan(env)
        b, _ = RRTPlanner(seed=5, sampler=UniformSampler()).plan(env)
        self.assertEqual(a, b)

    def test_name_override(self):
        p = RRTStarPlanner(name="RRT* + learned")
        _, m = p.plan(get_scenario("empty"))
        self.assertEqual(m["planner"], "RRT* + learned")
        self.assertEqual(RRTStarPlanner().name, "RRT*")   # class default untouched


class TestHeatmapSampler(unittest.TestCase):

    def setUp(self):
        self.env = Environment(100, 100, [], (5, 5), (95, 95), robot_radius=1.0)

    def test_samples_land_in_weighted_cells(self):
        heat = np.zeros((10, 10))
        heat[7, 2] = 1.0                    # row 7 -> y in [70, 80); col 2 -> x in [20, 30)
        s = HeatmapSampler(heat, mix=1.0)
        rng = random.Random(0)
        for _ in range(200):
            x, y = s.sample(self.env, rng)
            self.assertTrue(20 <= x <= 30 and 70 <= y <= 80, (x, y))

    def test_mix_fraction(self):
        heat = np.zeros((10, 10))
        heat[0, 0] = 1.0                    # the cell [0,10) x [0,10)
        s = HeatmapSampler(heat, mix=0.3)
        rng = random.Random(1)
        n = 20000
        inside = sum(1 for _ in range(n)
                     if (lambda p: p[0] < 10 and p[1] < 10)(s.sample(self.env, rng)))
        # Expected: 0.3 from the heatmap + 0.7 * (area share of that cell).
        expected = 0.3 + 0.7 * (9.0 / 98.0) ** 2
        self.assertAlmostEqual(inside / n, expected, delta=0.015)

    def test_zero_heatmap_falls_back_to_uniform(self):
        s = HeatmapSampler(np.zeros((8, 8)), mix=1.0)
        rng = random.Random(2)
        pts = [s.sample(self.env, rng) for _ in range(500)]
        self.assertGreater(np.std([p[0] for p in pts]), 20)

    def test_samples_respect_robot_radius_bounds(self):
        heat = np.zeros((10, 10))
        heat[0, 0] = heat[9, 9] = 1.0      # corner cells
        s = HeatmapSampler(heat, mix=1.0)
        rng = random.Random(3)
        for _ in range(500):
            self.assertTrue(self.env.in_bounds(s.sample(self.env, rng)))

    def test_misleading_heatmap_still_gives_valid_path(self):
        # A heatmap pointing to the wrong corner: the uniform share must still
        # let RRT* find a valid path (the completeness argument in samplers.py).
        env = get_scenario("narrow_gap")
        heat = np.zeros((64, 64))
        heat[60:, :4] = 1.0
        path, m = RRTStarPlanner(seed=0, max_iters=6000,
                                 sampler=HeatmapSampler(heat, mix=0.5)).plan(env)
        self.assertTrue(m["success"])
        self.assertTrue(verify_path(env, path)[0])


@unittest.skipUnless(HAVE_SCIPY, "SciPy not installed")
class TestLabels(unittest.TestCase):

    def test_empty_world_label_is_band_around_line(self):
        from learning.labels import near_optimal_region
        free = np.ones((64, 64), dtype=bool)
        label, d_opt = near_optimal_region(free, (5, 32), (58, 32))
        self.assertAlmostEqual(d_opt, 53.0)
        # The straight line is inside the label ...
        self.assertTrue(label[32, 5:59].all())
        # ... far-away cells are not.
        self.assertFalse(label[5, 32] or label[60, 32])

    def test_optimal_route_is_labelled_through_doorway(self):
        from learning.labels import near_optimal_region
        free = np.ones((64, 64), dtype=bool)
        free[:, 30:34] = False
        free[40:44, 30:34] = True           # doorway in the wall
        label, _ = near_optimal_region(free, (10, 10), (54, 10))
        self.assertTrue(label[40:44, 30:34].any())
        self.assertTrue(label[free].sum() < free.sum())

    def test_unreachable_returns_none(self):
        from learning.labels import near_optimal_region
        free = np.ones((32, 32), dtype=bool)
        free[:, 15] = False
        label, d_opt = near_optimal_region(free, (2, 2), (30, 2))
        self.assertIsNone(label)
        self.assertTrue(math.isinf(d_opt))

    def test_splits_do_not_share_worlds(self):
        from learning.dataset import make_problem
        seeds = {split: {make_problem(split, i)["seed"] for i in range(30)}
                 for split in ("train", "val", "test")}
        self.assertFalse(seeds["train"] & seeds["val"])
        self.assertFalse(seeds["train"] & seeds["test"])
        self.assertFalse(seeds["val"] & seeds["test"])


@unittest.skipUnless(HAVE_TORCH and HAVE_SCIPY, "PyTorch / SciPy not installed")
class TestLearnedSampler(unittest.TestCase):

    def test_untrained_model_still_plans_validly(self):
        # Even random weights must not break the planner -- only slow it down.
        from learning.model import SamplingUNet
        from learning.sampler import LearnedSampler
        torch.manual_seed(0)
        env = get_scenario("bug_trap")
        sampler = LearnedSampler(SamplingUNet(), mix=0.5)
        path, m = RRTStarPlanner(seed=1, max_iters=6000, sampler=sampler).plan(env)
        self.assertTrue(m["success"])
        self.assertTrue(verify_path(env, path)[0])
        self.assertEqual(sampler.last_probabilities.shape, (64, 64))
        # Blocked cells get zero weight.
        from learning.features import free_mask
        self.assertTrue((sampler.last_probabilities[~free_mask(env)] == 0).all())

    def test_inference_time_counted_in_planner_time(self):
        from learning.model import SamplingUNet
        from learning.sampler import LearnedSampler
        sampler = LearnedSampler(SamplingUNet(), mix=0.5)
        _, m = RRTPlanner(seed=0, sampler=sampler).plan(get_scenario("empty"))
        self.assertGreaterEqual(m["time_sec"], sampler.last_prepare_time)

    def test_model_output_shape(self):
        from learning.features import problem_features
        from learning.model import SamplingUNet
        x, _ = problem_features(get_scenario("maze"))
        with torch.no_grad():
            y = SamplingUNet().eval()(torch.from_numpy(x)[None])
        self.assertEqual(tuple(y.shape), (1, 64, 64))

    def test_dijkstra_sampler_concentrates_in_label(self):
        from learning.sampler import DijkstraSampler
        env = Environment(100, 100, [RectObstacle(45, 0, 10, 100)], (5, 50), (30, 50),
                          robot_radius=1.0).validate()
        s = DijkstraSampler(mix=1.0)
        s.prepare(env, env.start, env.goal)
        rng = random.Random(0)
        xs = [s.sample(env, rng)[0] for _ in range(300)]
        self.assertTrue(max(xs) < 45)       # nothing beyond the wall


if __name__ == "__main__":
    unittest.main()
