"""
Tests for the arm's learned sampler (learning/arm_*.py).

What these protect:

  * The mirror augmentation used in training is a TRUE symmetry: reflecting
    the world through the arm's base and negating every joint angle gives
    exactly the same collision answers. If it were not, augmentation would
    teach the network wrong poses.
  * The obstacle image and the path resampling do what training assumes.
  * The sampler returns valid joint vectors, and even an untrained network
    cannot stop the planner from finding a verified path (the uniform share
    keeps the guarantees).

Skipped automatically when PyTorch is not installed.
"""

import math
import random
import unittest

import numpy as np

from arm import ArmEnvironment, random_arm_problem, scenario_two_link
from env.obstacles import CircleObstacle, RectObstacle
from planners import InformedSampler, RRTStarPlanner, verify_path

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _mirror_env(env):
    """Reflect every obstacle through the horizontal line y = base_y."""
    by = float(env.arm.base[1])
    obs = []
    for o in env.obstacles:
        if isinstance(o, CircleObstacle):
            obs.append(CircleObstacle(o.cx, 2 * by - o.cy, o.r))
        else:
            obs.append(RectObstacle(o.x, 2 * by - o.y - o.h, o.w, o.h))
    return ArmEnvironment(env.arm, obs, tuple(-v for v in env.start),
                          tuple(-v for v in env.goal))


class TestMirrorSymmetry(unittest.TestCase):

    def test_mirror_world_equals_negated_angles(self):
        for dof, seed in ((2, 1), (4, 5), (7, 9)):
            env = random_arm_problem(dof, 40_000_000 + seed)
            mir = _mirror_env(env)
            Q = np.random.default_rng(seed).uniform(-math.pi, math.pi, (3000, dof))
            np.testing.assert_array_equal(env._collides(Q), mir._collides(-Q))

    def test_occupancy_image_flips_with_the_world(self):
        from learning.arm_data import occupancy_image
        env = random_arm_problem(4, 40_000_123)
        np.testing.assert_array_equal(occupancy_image(_mirror_env(env)),
                                      occupancy_image(env)[::-1])


class TestDataHelpers(unittest.TestCase):

    def test_occupancy_image(self):
        from learning.arm_data import occupancy_image
        env = ArmEnvironment(scenario_two_link().arm, [RectObstacle(0, 0, 50, 100)],
                             (0.0, 0.0), (1.0, 0.0))
        occ = occupancy_image(env, 32)
        self.assertTrue(occ[:, :16].all())
        self.assertFalse(occ[:, 16:].any())

    def test_resample_path(self):
        from learning.arm_data import resample_path
        path = [(0.0, 0.0), (1.0, 0.0), (1.0, 3.0)]          # total length 4
        P = resample_path(path, 5)
        np.testing.assert_allclose(P[0], path[0])
        np.testing.assert_allclose(P[-1], path[-1])
        np.testing.assert_allclose(P[1], (1.0, 0.0))         # 1/4 of the way
        steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
        np.testing.assert_allclose(steps, 1.0)                # evenly spaced

    def test_point_to_path(self):
        from learning.arm_train import point_to_path
        P = np.array([[0, 0], [1, 0], [1, 1]], float)
        d = point_to_path(np.array([[0.5, 0], [1, 0.5], [2, 0], [0.5, 0.5]]), P)
        np.testing.assert_allclose(d, [0, 0, 1, 0.5])


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TestArmSampler(unittest.TestCase):

    def setUp(self):
        from learning.arm_model import ArmCVAE
        torch.manual_seed(0)
        self.model = ArmCVAE(2).eval()

    def test_samples_are_valid_joint_vectors(self):
        from learning.arm_sampler import ArmLearnedSampler
        env = scenario_two_link()
        s = ArmLearnedSampler(self.model, mix=1.0, batch=64)
        s.prepare(env, env.start, env.goal)
        rng = random.Random(0)
        for _ in range(200):                                  # crosses a refill
            q = s.sample(env, rng)
            self.assertEqual(len(q), 2)
            self.assertTrue(env.in_bounds(q))

    def test_reproducible_with_same_seed(self):
        from learning.arm_sampler import ArmLearnedSampler
        env = scenario_two_link()
        a, b = (ArmLearnedSampler(self.model, mix=1.0, seed=3) for _ in range(2))
        a.prepare(env, env.start, env.goal)
        b.prepare(env, env.start, env.goal)
        r1, r2 = random.Random(1), random.Random(1)
        self.assertEqual([a.sample(env, r1) for _ in range(20)],
                         [b.sample(env, r2) for _ in range(20)])

    def test_untrained_model_still_plans_validly(self):
        from learning.arm_sampler import ArmLearnedSampler
        env = scenario_two_link()
        for sampler in (ArmLearnedSampler(self.model, mix=0.5),
                        InformedSampler(base=ArmLearnedSampler(self.model, mix=0.5))):
            path, m = RRTStarPlanner(step_size=0.5, goal_tolerance=0.5, seed=0,
                                     max_iters=3000, sampler=sampler).plan(env)
            self.assertTrue(m["success"])
            self.assertTrue(verify_path(env, path)[0])


if __name__ == "__main__":
    unittest.main()
