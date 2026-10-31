"""
Obstacle primitives: axis-aligned rectangles and circles.

Both expose the same two queries the planners need:

    clearance_to_point(p)     -> distance from p to the obstacle surface
    clearance_to_segment(a,b) -> distance from segment a-b to the surface

A configuration is in collision when the clearance is <= the robot radius.
Phrasing collision as a *distance* (rather than a boolean overlap test) is
what lets us model the robot as a disc for free: inflating the obstacle by
the robot radius is the Minkowski sum, and comparing distance <= radius is
exactly equivalent to it.
"""

from . import geometry as geo


class Obstacle:
    """Interface shared by all obstacle shapes."""

    def clearance_to_point(self, p):
        raise NotImplementedError

    def clearance_to_segment(self, a, b):
        raise NotImplementedError

    def bounding_box(self):
        """(xmin, ymin, xmax, ymax) enclosing the obstacle."""
        raise NotImplementedError

    def to_dict(self):
        """Serialisable form, so environments can be logged / reproduced."""
        raise NotImplementedError


class RectObstacle(Obstacle):
    """Axis-aligned rectangle with lower-left corner (x, y) and size w x h."""

    kind = "rect"

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = float(x), float(y), float(w), float(h)

    @property
    def box(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def clearance_to_point(self, p):
        return geo.point_aabb_distance(p, self.box)

    def clearance_to_segment(self, a, b):
        return geo.segment_aabb_distance(a, b, self.box)

    def bounding_box(self):
        return self.box

    def to_dict(self):
        return {"kind": "rect", "x": self.x, "y": self.y, "w": self.w, "h": self.h}


class CircleObstacle(Obstacle):
    """Disc obstacle centred at (cx, cy) with radius r."""

    kind = "circle"

    def __init__(self, cx, cy, r):
        self.cx, self.cy, self.r = float(cx), float(cy), float(r)

    @property
    def center(self):
        return (self.cx, self.cy)

    def clearance_to_point(self, p):
        # Negative inside the disc; clamped at 0 so "touching" reads as 0.
        return max(0.0, geo.distance(p, self.center) - self.r)

    def clearance_to_segment(self, a, b):
        return max(0.0, geo.point_segment_distance(self.center, a, b) - self.r)

    def bounding_box(self):
        return (self.cx - self.r, self.cy - self.r, self.cx + self.r, self.cy + self.r)

    def to_dict(self):
        return {"kind": "circle", "cx": self.cx, "cy": self.cy, "r": self.r}


def obstacle_from_dict(d):
    """Rebuild an obstacle from its to_dict() form."""
    if d["kind"] == "rect":
        return RectObstacle(d["x"], d["y"], d["w"], d["h"])
    if d["kind"] == "circle":
        return CircleObstacle(d["cx"], d["cy"], d["r"])
    raise ValueError("unknown obstacle kind: %r" % (d["kind"],))
