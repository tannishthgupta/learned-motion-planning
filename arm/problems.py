"""
Arm planning problems: fixed scenarios and random instances.

All arms share the same total reach (40 units) whatever their number of
joints, so a 2-joint and a 7-joint arm can reach the same places and the
difference between them is purely the dimension of joint space.
"""

import math
import random

import numpy as np

from env.obstacles import CircleObstacle, RectObstacle

from .environment import JOINT_LIMIT, ArmEnvironment
from .robot import PlanarArm

REACH = 40.0
BASE = (50.0, 50.0)


def make_arm(dof, reach=REACH, base=BASE, link_radius=1.0):
    """An arm with `dof` equal links and the given total reach."""
    return PlanarArm([reach / dof] * dof, base=base, link_radius=link_radius)


# ----------------------------------------------------------------------------
# Fixed scenarios
# ----------------------------------------------------------------------------

def scenario_two_link():
    """2-joint arm swinging from pointing right to pointing left.

    Two obstacles block the short ways round (straight up and straight down
    are both partly blocked), so the arm has to fold its elbow to squeeze
    past. Small enough (2 joints) that the whole joint space can be drawn as
    an image: the classic configuration-space figure.
    """
    arm = PlanarArm([22.0, 18.0], base=BASE, link_radius=1.0)
    obstacles = [
        CircleObstacle(50.0, 82.0, 7.0),          # above the base
        RectObstacle(26.0, 14.0, 12.0, 12.0),     # below-left
        RectObstacle(70.0, 28.0, 8.0, 8.0),       # below-right
    ]
    start = (0.0, 0.0)                           # pointing right
    goal = (math.pi - 0.05, 0.0)                 # pointing left
    return ArmEnvironment(arm, obstacles, start, goal, name="two_link").validate()


SCENARIOS = {"two_link": scenario_two_link}


# ----------------------------------------------------------------------------
# Random problems
# ----------------------------------------------------------------------------

def random_arm_problem(dof, seed, n_obstacles=6, min_separation=2.0, max_attempts=200):
    """A random world around the arm plus a random start and goal.

    Obstacles are scattered in a ring around the base (never on top of it, or
    the arm could not move at all). Start and goal are random collision-free
    configurations at least `min_separation` radians apart, and the straight
    joint-space line between them must be blocked, so the planner has
    something to do.
    """
    rng = random.Random(seed)
    arm = make_arm(dof)
    for _ in range(max_attempts):
        obstacles = []
        while len(obstacles) < n_obstacles:
            ang = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(12.0, REACH + 4.0)
            cx = BASE[0] + dist * math.cos(ang)
            cy = BASE[1] + dist * math.sin(ang)
            size = rng.uniform(4.0, 9.0)
            if rng.random() < 0.5:
                obstacles.append(CircleObstacle(cx, cy, size / 2.0))
            else:
                obstacles.append(RectObstacle(cx - size / 2.0, cy - size / 2.0, size, size))

        env = ArmEnvironment(arm, obstacles, (0.0,) * dof, (0.0,) * dof,
                             name="arm%d_%d" % (dof, seed))
        margin = env.motion_resolution
        endpoints = []
        for _ in range(500):
            q = tuple(rng.uniform(-JOINT_LIMIT, JOINT_LIMIT) for _ in range(dof))
            if not env._collides(np.asarray(q)[None, :], margin=margin)[0]:
                endpoints.append(q)
                if len(endpoints) == 2:
                    break
        if len(endpoints) < 2:
            continue
        start, goal = endpoints
        if math.dist(start, goal) < min_separation:
            continue
        env = ArmEnvironment(arm, obstacles, start, goal, name="arm%d_%d" % (dof, seed))
        if env.is_motion_valid(start, goal):
            continue          # trivial: the straight line already works
        return env.validate()
    raise RuntimeError("no random arm problem for dof=%d seed=%d" % (dof, seed))
