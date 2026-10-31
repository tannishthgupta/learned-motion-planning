"""
The simulated 2D world.

A bounded rectangular workspace containing static obstacles, a start pose and
a goal pose. The robot is a disc of radius `robot_radius` (set it to 0 for a
point robot). Planning happens directly in (x, y) -- there is no orientation
state, no dynamics, and no physics engine, which is a deliberate scope choice.

The Environment is the *single* source of truth about feasibility. Every
planner asks it the same two questions:

    is_state_valid(p)        -- may the robot sit at p?
    is_motion_valid(a, b)    -- may the robot travel the straight line a->b?

Because `is_motion_valid` uses exact segment-to-obstacle distance rather than
sampling points along the edge, it cannot "tunnel" through a thin obstacle.
That matters for the comparison: a sampled checker would silently favour the
sampling-based planners, which take far longer steps than A*'s grid moves.
"""

import json
import math

from . import geometry as geo
from .obstacles import CircleObstacle, RectObstacle, obstacle_from_dict


class Environment:
    def __init__(self, width, height, obstacles=None, start=(0.0, 0.0),
                 goal=(0.0, 0.0), robot_radius=0.0, name="env"):
        self.width = float(width)
        self.height = float(height)
        self.obstacles = list(obstacles or [])
        self.start = (float(start[0]), float(start[1]))
        self.goal = (float(goal[0]), float(goal[1]))
        self.robot_radius = float(robot_radius)
        self.name = name

        # Broad-phase cache: one axis-aligned bounding box per obstacle.
        # Rebuild via refresh_cache() if the obstacle list is edited in place.
        self._boxes = [o.bounding_box() for o in self.obstacles]

        # Instrumentation. The benchmark harness reports these alongside
        # wall-clock time, because collision checking dominates runtime and
        # counting calls is machine-independent in a way that seconds are not.
        self.state_checks = 0
        self.motion_checks = 0

    # ------------------------------------------------------------------
    # Feasibility queries -- the interface every planner shares
    # ------------------------------------------------------------------

    def in_bounds(self, p):
        """True if the whole robot disc fits inside the workspace at p."""
        r = self.robot_radius
        return (r <= p[0] <= self.width - r) and (r <= p[1] <= self.height - r)

    def clearance(self, p):
        """Distance from p to the nearest obstacle surface (ignores bounds)."""
        if not self.obstacles:
            return float("inf")
        return min(o.clearance_to_point(p) for o in self.obstacles)

    def is_state_valid(self, p):
        """True if the robot can occupy point p without collision."""
        self.state_checks += 1
        if not self.in_bounds(p):
            return False

        r = self.robot_radius
        px, py = p
        for o, (oxmin, oymin, oxmax, oymax) in zip(self.obstacles, self._boxes):
            # Broad phase: reject on the bounding box before running the exact
            # test. If the boxes are separated along any axis by more than r,
            # the true distance also exceeds r, so this never rejects a real
            # collision -- it is a speed-up, not an approximation.
            if (oxmax < px - r or oxmin > px + r or
                    oymax < py - r or oymin > py + r):
                continue
            if o.clearance_to_point(p) <= r:
                return False
        return True

    def is_motion_valid(self, a, b):
        """True if the robot can sweep the straight segment a->b collision-free.

        Exact: the swept area of a disc along a segment is a capsule, and the
        capsule misses an obstacle exactly when the segment's distance to that
        obstacle exceeds the robot radius.
        """
        self.motion_checks += 1

        # Bounds first -- the workspace is convex, so checking the two
        # endpoints is sufficient for the entire segment.
        if not self.in_bounds(a) or not self.in_bounds(b):
            return False

        r = self.robot_radius

        # Broad phase: the swept capsule is contained in the segment's
        # bounding box inflated by r. An obstacle whose own box misses that
        # cannot be hit, so the exact (and much costlier) distance test is
        # skipped. Exact rejection, no false negatives.
        sxmin = (a[0] if a[0] < b[0] else b[0]) - r
        sxmax = (a[0] if a[0] > b[0] else b[0]) + r
        symin = (a[1] if a[1] < b[1] else b[1]) - r
        symax = (a[1] if a[1] > b[1] else b[1]) + r

        for o, (oxmin, oymin, oxmax, oymax) in zip(self.obstacles, self._boxes):
            if (oxmax < sxmin or oxmin > sxmax or
                    oymax < symin or oymin > symax):
                continue
            if o.clearance_to_segment(a, b) <= r:
                return False
        return True

    def refresh_cache(self):
        """Recompute the broad-phase boxes after editing `obstacles` in place."""
        self._boxes = [o.bounding_box() for o in self.obstacles]
        return self

    def reset_counters(self):
        self.state_checks = 0
        self.motion_checks = 0

    # ------------------------------------------------------------------
    # Sampling (used by RRT / RRT*)
    # ------------------------------------------------------------------

    def sample(self, rng):
        """Uniform sample from the workspace, ignoring obstacles."""
        r = self.robot_radius
        return (rng.uniform(r, self.width - r), rng.uniform(r, self.height - r))

    def sample_free(self, rng, max_tries=1000):
        """Uniform sample from the *free* workspace by rejection sampling."""
        for _ in range(max_tries):
            p = self.sample(rng)
            if self.is_state_valid(p):
                return p
        raise RuntimeError("could not find a collision-free sample in %d tries; "
                           "the environment may be over-populated" % max_tries)

    # ------------------------------------------------------------------
    # Validation and bookkeeping
    # ------------------------------------------------------------------

    def validate(self):
        """Raise if the problem instance is malformed. Returns self.

        A planner failing on an invalid instance would pollute the success-rate
        metric, so every generated environment is validated up front.
        """
        if not self.is_state_valid(self.start):
            raise ValueError("start %s is in collision or out of bounds" % (self.start,))
        if not self.is_state_valid(self.goal):
            raise ValueError("goal %s is in collision or out of bounds" % (self.goal,))
        if geo.distance(self.start, self.goal) < 1e-9:
            raise ValueError("start and goal are identical")
        self.reset_counters()
        return self

    def obstacle_area(self):
        """Total area covered by obstacles (overlaps counted twice)."""
        total = 0.0
        for o in self.obstacles:
            if isinstance(o, RectObstacle):
                total += o.w * o.h
            elif isinstance(o, CircleObstacle):
                total += math.pi * o.r ** 2
        return total

    def obstacle_density(self):
        """Nominal fraction of the workspace occupied by obstacles."""
        return self.obstacle_area() / (self.width * self.height)

    # ------------------------------------------------------------------
    # Serialisation -- lets the benchmark replay the exact same world set
    # ------------------------------------------------------------------

    def to_dict(self):
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "robot_radius": self.robot_radius,
            "start": list(self.start),
            "goal": list(self.goal),
            "obstacles": [o.to_dict() for o in self.obstacles],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            width=d["width"], height=d["height"],
            obstacles=[obstacle_from_dict(o) for o in d["obstacles"]],
            start=tuple(d["start"]), goal=tuple(d["goal"]),
            robot_radius=d.get("robot_radius", 0.0),
            name=d.get("name", "env"),
        )

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __repr__(self):
        return ("Environment(name=%r, %gx%g, %d obstacles, density=%.2f, "
                "start=%s, goal=%s)" % (self.name, self.width, self.height,
                                        len(self.obstacles), self.obstacle_density(),
                                        self.start, self.goal))
