"""
Tests for the planar robot arm (arm/) and N-D planning.

What these protect:

  * Forward kinematics puts the links where the geometry says they are.
  * The fast vectorised collision geometry agrees with the scalar
    env/geometry.py functions it re-implements.
  * Collision checks catch link-vs-obstacle and link-vs-link contact.
  * is_motion_valid is SOUND: when it says a motion is free, a much finer
    check of the same motion never finds a collision. This is the property
    the docstring of arm/environment.py argues for.
  * RRT, RRT* and Informed RRT* plan for the arm without modification, and
    every path they return is verified collision-free.
"""

import math
import random
import unittest

import numpy as np

from arm import ArmEnvironment, PlanarArm, make_arm, random_arm_problem, scenario_two_link
from arm import collision as col
from env import geometry as geo
from env.obstacles import CircleObstacle, RectObstacle
from planners import (InformedSampler, RRTPlanner, RRTStarPlanner, Tree, verify_path)
from planners.rrt import steer


class TestKinematics(unittest.TestCase):

    def test_straight_arm(self):
        arm = PlanarArm([10, 5], base=(0, 0))
        pts = arm.joint_positions((0.0, 0.0))
        np.testing.assert_allclose(pts, [[0, 0], [10, 0], [15, 0]], atol=1e-12)

    def test_relative_angles(self):
        # Second joint angle is relative to the first link.
        arm = PlanarArm([10, 5], base=(1, 2))
        pts = arm.joint_positions((math.pi / 2, -math.pi / 2))
        np.testing.assert_allclose(pts, [[1, 2], [1, 12], [6, 12]], atol=1e-12)

    def test_batch_matches_single(self):
        arm = make_arm(5)
        rng = np.random.default_rng(0)
        Q = rng.uniform(-3, 3, size=(20, 5))
        batch = arm.joint_positions_batch(Q)
        for k in range(20):
            np.testing.assert_allclose(batch[k], arm.joint_positions(Q[k]))

    def test_reach_from(self):
        arm = PlanarArm([3, 2, 1])
        np.testing.assert_allclose(arm.reach_from, [6, 3, 1])


class TestVectorisedGeometry(unittest.TestCase):
    """Each fast function must match its scalar twin in env/geometry.py."""

    def setUp(self):
        rng = np.random.default_rng(1)
        self.A = rng.uniform(0, 20, size=(400, 2))
        self.B = rng.uniform(0, 20, size=(400, 2))
        # Include degenerate and axis-parallel segments.
        self.B[:20] = self.A[:20]
        self.B[20:40, 0] = self.A[20:40, 0]
        self.B[40:60, 1] = self.A[40:60, 1]

    def test_segment_box(self):
        box = (6.0, 7.0, 11.0, 9.5)
        d = col.segment_box_distance(self.A, self.B, box)
        for k in range(len(self.A)):
            self.assertAlmostEqual(d[k], geo.segment_aabb_distance(self.A[k], self.B[k], box), 9)

    def test_segment_circle(self):
        d = col.segment_circle_distance(self.A, self.B, (10.0, 10.0), 3.0)
        c = CircleObstacle(10.0, 10.0, 3.0)
        for k in range(len(self.A)):
            self.assertAlmostEqual(d[k], c.clearance_to_segment(self.A[k], self.B[k]), 9)

    def test_segment_segment(self):
        C, D = self.A[::-1].copy(), self.B[::-1].copy()
        d = col.segment_segment_distance(self.A, self.B, C, D)
        for k in range(len(self.A)):
            self.assertAlmostEqual(
                d[k], geo.segment_segment_distance(self.A[k], self.B[k], C[k], D[k]), 9)


class TestArmCollision(unittest.TestCase):

    def test_link_hits_obstacle(self):
        arm = PlanarArm([10, 10], base=(0, 0), link_radius=0.5)
        env = ArmEnvironment(arm, [CircleObstacle(15, 0, 1)], (0, 0), (1, 0))
        self.assertFalse(env.is_state_valid((0.0, 0.0)))        # pointing straight at it
        self.assertTrue(env.is_state_valid((math.pi / 2, 0.0)))  # pointing away

    def test_self_collision(self):
        # A 3-link arm folded back so link 3 crosses link 1: link 1 runs
        # (0,0)->(10,0), link 2 goes up to (10,3), link 3 heads down-left
        # and cuts through link 1 near x = 3.
        arm = PlanarArm([10, 3, 10], base=(0, 0), link_radius=0.2)
        env = ArmEnvironment(arm, [], (0, 0, 0), (0, 0, 0))
        folded = (0.0, math.pi / 2, 2.0)
        self.assertFalse(env.is_state_valid(folded))
        env.self_collision = False
        self.assertTrue(env.is_state_valid(folded))

    def test_joint_limits(self):
        env = ArmEnvironment(make_arm(2), [], (0, 0), (1, 1))
        self.assertFalse(env.is_state_valid((3.2, 0.0)))
        self.assertFalse(env.is_motion_valid((0.0, 0.0), (3.2, 0.0)))

    def test_motion_check_is_sound(self):
        # Whenever is_motion_valid says yes, checking the same motion at a
        # resolution 20x finer (with no safety margin) must also say yes.
        env = random_arm_problem(4, seed=3)
        rng = random.Random(0)
        accepted = 0
        for _ in range(400):
            qa = env.sample(rng)
            qb = steer(qa, env.sample(rng), 0.8)
            if not (env.is_state_valid(qa) and env.is_motion_valid(qa, qb)):
                continue
            accepted += 1
            n = 20 * env.motion_steps(qa, qb)
            t = np.linspace(0, 1, n + 1)[:, None]
            Q = (1 - t) * np.asarray(qa) + t * np.asarray(qb)
            self.assertFalse(env._collides(Q).any())
        self.assertGreater(accepted, 30)

    def test_thin_wall_not_tunnelled(self):
        # A wide sweep past a thin wall: the coarse-looking motion must be rejected.
        arm = PlanarArm([20], base=(0, 0), link_radius=0.1)
        env = ArmEnvironment(arm, [RectObstacle(5, -0.05, 20, 0.1)], (-1.0,), (1.0,))
        self.assertFalse(env.is_motion_valid((-1.0,), (1.0,)))


class TestNDPlanning(unittest.TestCase):

    def test_tree_and_steer_in_5d(self):
        t = Tree((0.0,) * 5)
        i = t.add_node((1.0, 0, 0, 0, 0), parent=0, cost=1.0)
        self.assertEqual(t.point(i), (1.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(t.nearest((0.9, 0, 0, 0, 0)), i)
        p = steer((0.0,) * 5, (3.0, 4.0, 0, 0, 0), 1.0)
        self.assertAlmostEqual(math.dist(p, (0.0,) * 5), 1.0)

    def test_planners_on_two_link_arm(self):
        env = scenario_two_link()
        for planner in (RRTPlanner(step_size=0.5, goal_tolerance=0.5, seed=0, max_iters=4000),
                        RRTStarPlanner(step_size=0.5, goal_tolerance=0.5, seed=0, max_iters=800),
                        RRTStarPlanner(step_size=0.5, goal_tolerance=0.5, seed=0, max_iters=800,
                                       sampler=InformedSampler())):
            path, m = planner.plan(env)
            self.assertTrue(m["success"], planner)
            self.assertTrue(verify_path(env, path)[0], planner)
            self.assertEqual(len(path[0]), 2)

    def test_informed_ellipsoid_in_4d(self):
        env = ArmEnvironment(make_arm(4), [], (0.0, 0.5, -0.5, 0.0), (1.0, 1.0, 0.0, -1.0))
        s = InformedSampler()
        s.prepare(env, env.start, env.goal)
        c = 1.3 * math.dist(env.start, env.goal)
        s.notify_best_cost(c)
        rng = random.Random(0)
        for _ in range(500):
            p = s.sample(env, rng)
            self.assertEqual(len(p), 4)
            self.assertLessEqual(math.dist(p, env.start) + math.dist(p, env.goal), c + 1e-9)


if __name__ == "__main__":
    unittest.main()
