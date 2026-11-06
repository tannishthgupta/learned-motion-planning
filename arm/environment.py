"""
The arm's planning problem, in the same shape as env.Environment.

RRT, RRT* and Informed RRT* only ever ask an environment a few questions:

    sample(rng)                  a random configuration
    in_bounds(q)                 is q inside the joint limits?
    is_state_valid(q)            can the arm be at q without hitting anything?
    is_motion_valid(qa, qb)      can it move from qa to qb (straight line in
                                 joint space) without hitting anything?
    free_volume(), dimension     used by RRT* to size its rewiring ball

ArmEnvironment answers them for a planar N-joint arm, so the *unchanged*
planners plan for the arm. A configuration is a tuple of N joint angles in
radians, and "distance" between configurations is the Euclidean norm of the
angle differences: how much the joints turn in total. That is the cost the
planners minimise.

Joint limits are [-pi, pi] with no wrap-around, like a real joint with stops.

Motion checking: sound, not sampled at a guess
----------------------------------------------
For the 2-D robot, is_motion_valid is exact. For the arm it cannot be (the
links sweep curved regions), so it checks a series of intermediate poses.
The usual shortcut -- "check every 0.05 rad and hope" -- can let a link pass
straight through a thin obstacle between two checks. This implementation
rules that out:

  1. When the joints move by dq along the straight line, no point of the arm
     moves further than  D = sum_i |dq_i| * R_i,  where R_i is the length
     from joint i to the tip (a point at distance R from a turning joint moves
     at most R times the angle).
  2. Split the motion into n = ceil(D / delta) equal steps. Every in-between
     pose is then within delta / 2 of a checked pose, for every point on the
     arm.
  3. At the checked poses, demand clearance of  link_radius + delta / 2
     instead of just link_radius.

So if every checked pose passes, every in-between pose has at least
link_radius of clearance -- the motion is guaranteed collision-free. The cost
is that the check is slightly conservative near obstacles. This mirrors the
2-D environment's design choice of exact checking over sampled checking.
"""

import math

import numpy as np

from env.obstacles import CircleObstacle, RectObstacle, obstacle_from_dict

from . import collision as col
from .robot import PlanarArm

JOINT_LIMIT = math.pi


class ArmEnvironment:
    """
    Parameters
    ----------
    arm : PlanarArm
    obstacles : list of RectObstacle / CircleObstacle (world coordinates)
    start, goal : sequences of N joint angles
    width, height : size of the world, for drawing only
    motion_resolution : delta above, in world units. Smaller = less
        conservative near obstacles, but more poses to check per motion.
    self_collision : check non-neighbouring links against each other
    """

    def __init__(self, arm, obstacles, start, goal, width=100.0, height=100.0,
                 motion_resolution=1.0, self_collision=True, name="arm"):
        self.arm = arm
        self.obstacles = list(obstacles)
        self.start = tuple(float(v) for v in start)
        self.goal = tuple(float(v) for v in goal)
        if len(self.start) != arm.dof or len(self.goal) != arm.dof:
            raise ValueError("start/goal must have one angle per joint (%d)" % arm.dof)
        self.width = float(width)
        self.height = float(height)
        self.motion_resolution = float(motion_resolution)
        self.self_collision = bool(self_collision)
        self.name = name

        # Pairs of links that are not neighbours (neighbours share a joint and
        # always "touch"), for the self-collision check.
        n = arm.dof
        self._pairs = np.array([(i, j) for i in range(n) for j in range(i + 2, n)],
                               dtype=int).reshape(-1, 2)

        self.state_checks = 0
        self.motion_checks = 0
        self._free_fraction = None

    @property
    def dimension(self):
        return self.arm.dof

    # ------------------------------------------------------------------
    # The core test: do these poses collide, with a given safety margin?
    # ------------------------------------------------------------------

    def _collides(self, Q, margin=0.0):
        """For each pose in Q (M, N), True if the arm collides there.

        `margin` is added to the link radius (see the module docstring).
        Uncounted: callers increment the public counters.
        """
        pts = self.arm.joint_positions_batch(Q)            # (M, N+1, 2)
        A, B = pts[:, :-1, :], pts[:, 1:, :]               # link segments (M, N, 2)
        clear = self.arm.link_radius + margin
        hit = np.zeros(Q.shape[0], dtype=bool)

        for o in self.obstacles:
            if isinstance(o, CircleObstacle):
                d = col.segment_circle_distance(A, B, (o.cx, o.cy), o.r)
            elif isinstance(o, RectObstacle):
                d = col.segment_box_distance(A, B, o.box)
            else:
                raise TypeError("unsupported obstacle %r" % type(o))
            hit |= (d <= clear).any(axis=1)

        if self.self_collision and len(self._pairs):
            i, j = self._pairs[:, 0], self._pairs[:, 1]
            d = col.segment_segment_distance(A[:, i], B[:, i], A[:, j], B[:, j])
            hit |= (d <= 2.0 * clear).any(axis=1)
        return hit

    # ------------------------------------------------------------------
    # Planner interface
    # ------------------------------------------------------------------

    def in_bounds(self, q):
        return all(-JOINT_LIMIT <= v <= JOINT_LIMIT for v in q)

    def sample(self, rng):
        return tuple(rng.uniform(-JOINT_LIMIT, JOINT_LIMIT) for _ in range(self.arm.dof))

    def is_state_valid(self, q):
        self.state_checks += 1
        if not self.in_bounds(q):
            return False
        return not self._collides(np.asarray(q, dtype=float)[None, :])[0]

    def motion_steps(self, qa, qb):
        """How many equal steps is_motion_valid splits qa -> qb into."""
        dq = np.abs(np.asarray(qb, float) - np.asarray(qa, float))
        sweep = float(np.dot(dq, self.arm.reach_from))    # bound D from the docstring
        return max(1, int(math.ceil(sweep / self.motion_resolution)))

    def is_motion_valid(self, qa, qb):
        self.motion_checks += 1
        # Joint limits form a box, which is convex: if both ends are inside,
        # the whole straight line is.
        if not self.in_bounds(qa) or not self.in_bounds(qb):
            return False
        n = self.motion_steps(qa, qb)
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        Q = (1.0 - t) * np.asarray(qa, float) + t * np.asarray(qb, float)
        return not self._collides(Q, margin=self.motion_resolution / 2.0).any()

    def reset_counters(self):
        self.state_checks = 0
        self.motion_checks = 0

    # ------------------------------------------------------------------
    # Size of the free space (RRT*'s rewiring radius needs it)
    # ------------------------------------------------------------------

    def free_fraction(self, n_samples=4000, seed=12345):
        """Fraction of joint space that is collision-free, by Monte Carlo.

        Fixed seed and cached, so it is the same every time; not counted as
        planner work.
        """
        if self._free_fraction is None:
            rng = np.random.default_rng(seed)
            Q = rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, size=(n_samples, self.arm.dof))
            self._free_fraction = float(1.0 - self._collides(Q).mean())
        return self._free_fraction

    def free_volume(self):
        """Collision-free volume of joint space (radians^N), floored at 5%."""
        return (2.0 * JOINT_LIMIT) ** self.arm.dof * max(0.05, self.free_fraction())

    # ------------------------------------------------------------------

    def validate(self, endpoint_margin=None):
        """Raise if start or goal is in collision. Returns self.

        Start and goal must also clear the motion-check margin, otherwise no
        motion could ever leave or reach them.
        """
        m = self.motion_resolution / 2.0 if endpoint_margin is None else endpoint_margin
        for label, q in (("start", self.start), ("goal", self.goal)):
            if not self.in_bounds(q):
                raise ValueError("%s %s is outside the joint limits" % (label, q))
            if self._collides(np.asarray(q)[None, :], margin=m)[0]:
                raise ValueError("%s %s is in collision" % (label, q))
        self.reset_counters()
        return self

    def to_dict(self):
        return {"name": self.name, "width": self.width, "height": self.height,
                "link_lengths": self.arm.link_lengths.tolist(),
                "base": self.arm.base.tolist(), "link_radius": self.arm.link_radius,
                "start": list(self.start), "goal": list(self.goal),
                "motion_resolution": self.motion_resolution,
                "obstacles": [o.to_dict() for o in self.obstacles]}

    @classmethod
    def from_dict(cls, d):
        arm = PlanarArm(d["link_lengths"], d["base"], d["link_radius"])
        return cls(arm, [obstacle_from_dict(o) for o in d["obstacles"]], d["start"],
                   d["goal"], d["width"], d["height"], d["motion_resolution"],
                   name=d.get("name", "arm"))

    def __repr__(self):
        return "ArmEnvironment(%r, dof=%d, %d obstacles)" % (
            self.name, self.arm.dof, len(self.obstacles))
