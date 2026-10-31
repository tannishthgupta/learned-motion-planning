"""
Exact 2D geometry primitives used by every collision check in the project.

Everything here is pure analytic geometry -- no physics engine, no external
solver. Each routine is written to be individually explainable, because the
collision model is what makes the planner comparison fair: all three planners
share these functions, so none of them gets an easier feasibility test.

Convention: a "point" is any (x, y) pair; a "segment" is a pair of points;
an "AABB" (axis-aligned bounding box) is the tuple (xmin, ymin, xmax, ymax).
"""

import math


# ----------------------------------------------------------------------------
# Basic predicates
# ----------------------------------------------------------------------------

def _orientation(a, b, c):
    """Twice the signed area of triangle (a, b, c).

    > 0  -> c lies left of the directed line a->b (counter-clockwise turn)
    < 0  -> c lies right of it (clockwise turn)
    = 0  -> a, b, c are collinear
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    """True if collinear point p lies inside the bounding box of segment a-b."""
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def segments_intersect(p1, p2, q1, q2):
    """Standard orientation test for proper or touching segment intersection."""
    d1 = _orientation(q1, q2, p1)
    d2 = _orientation(q1, q2, p2)
    d3 = _orientation(p1, p2, q1)
    d4 = _orientation(p1, p2, q2)

    # Proper crossing: each segment straddles the line through the other.
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True

    # Degenerate cases: an endpoint lies exactly on the other segment.
    if d1 == 0 and _on_segment(q1, q2, p1):
        return True
    if d2 == 0 and _on_segment(q1, q2, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, q1):
        return True
    if d4 == 0 and _on_segment(p1, p2, q2):
        return True
    return False


# ----------------------------------------------------------------------------
# Distances
# ----------------------------------------------------------------------------

def distance(p, q):
    """Euclidean distance between two points."""
    return math.hypot(q[0] - p[0], q[1] - p[1])


def point_segment_distance(p, a, b):
    """Shortest distance from point p to segment a-b.

    Projects p onto the infinite line a-b, clamps the projection parameter to
    [0, 1] so it stays on the segment, then measures to that closest point.
    """
    abx, aby = b[0] - a[0], b[1] - a[1]
    apx, apy = p[0] - a[0], p[1] - a[1]
    denom = abx * abx + aby * aby

    if denom == 0.0:              # degenerate segment (a == b)
        return math.hypot(apx, apy)

    t = (apx * abx + apy * aby) / denom
    t = max(0.0, min(1.0, t))     # clamp onto the segment
    cx, cy = a[0] + t * abx, a[1] + t * aby
    return math.hypot(p[0] - cx, p[1] - cy)


def segment_segment_distance(p1, p2, q1, q2):
    """Shortest distance between two segments (0 if they intersect).

    For non-intersecting segments the minimum is always attained at an
    endpoint of one segment, so checking all four endpoint-to-segment
    distances is exact.
    """
    if segments_intersect(p1, p2, q1, q2):
        return 0.0
    return min(
        point_segment_distance(p1, q1, q2),
        point_segment_distance(p2, q1, q2),
        point_segment_distance(q1, p1, p2),
        point_segment_distance(q2, p1, p2),
    )


def point_aabb_distance(p, box):
    """Shortest distance from a point to an axis-aligned box (0 if inside)."""
    xmin, ymin, xmax, ymax = box
    dx = max(xmin - p[0], 0.0, p[0] - xmax)
    dy = max(ymin - p[1], 0.0, p[1] - ymax)
    return math.hypot(dx, dy)


def segment_aabb_distance(p1, p2, box):
    """Shortest distance from a segment to an axis-aligned box (0 if it hits).

    If either endpoint is inside the box the distance is 0. Otherwise the
    segment lies entirely outside, and the closest approach is measured
    against the four edges of the box.
    """
    if point_aabb_distance(p1, box) == 0.0 or point_aabb_distance(p2, box) == 0.0:
        return 0.0

    xmin, ymin, xmax, ymax = box
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    best = float("inf")
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        best = min(best, segment_segment_distance(p1, p2, a, b))
        if best == 0.0:
            break
    return best
