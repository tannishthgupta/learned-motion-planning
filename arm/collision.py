"""
Vectorised segment geometry for arm collision checking.

These compute the same quantities as env/geometry.py (segment-to-circle and
segment-to-box distances, segment-to-segment distance), but for many segments
at once with NumPy. The arm needs this: checking one motion means checking
dozens of intermediate poses, each made of N link segments, against every
obstacle -- far too slow one segment at a time in plain Python.

The tests compare every function here against its scalar counterpart in
env/geometry.py on random inputs, so the two cannot drift apart.
"""

import numpy as np


def point_segment_distance(P, A, B):
    """Distance from points P to segments A-B. All (..., 2); broadcasts."""
    AB = B - A
    AP = P - A
    denom = np.einsum("...i,...i->...", AB, AB)
    t = np.where(denom > 0, np.einsum("...i,...i->...", AP, AB) / np.where(denom > 0, denom, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    diff = P - (A + t[..., None] * AB)
    return np.sqrt(np.einsum("...i,...i->...", diff, diff))


def segment_circle_distance(A, B, centre, radius):
    """Distance from segments A-B to a circle's surface (0 if touching/inside)."""
    return np.maximum(point_segment_distance(np.asarray(centre, float), A, B) - radius, 0.0)


def _point_box_distance(P, box):
    xmin, ymin, xmax, ymax = box
    dx = np.maximum(np.maximum(xmin - P[..., 0], 0.0), P[..., 0] - xmax)
    dy = np.maximum(np.maximum(ymin - P[..., 1], 0.0), P[..., 1] - ymax)
    return np.hypot(dx, dy)


def _segment_hits_box(A, B, box):
    """Liang-Barsky clipping: does segment A-B pass through the box?

    Parametrise the segment as A + t (B - A), t in [0, 1]. Each axis limits t
    to an interval where the point is between that axis's two box walls; the
    segment hits the box exactly when the intersection of those intervals
    with [0, 1] is non-empty.
    """
    xmin, ymin, xmax, ymax = box
    D = B - A
    # The box limits may be arrays (many boxes at once), so the result takes
    # the broadcast shape of segments and boxes.
    shape = np.broadcast_shapes(A.shape[:-1], np.shape(xmin))
    t_lo = np.zeros(shape)
    t_hi = np.ones(shape)
    ok = np.ones(shape, dtype=bool)
    for axis, lo, hi in ((0, xmin, xmax), (1, ymin, ymax)):
        a = A[..., axis]
        d = D[..., axis]
        parallel = d == 0.0
        # A segment parallel to this axis's walls is inside the slab or never.
        ok = ok & ~(parallel & ((a < lo) | (a > hi)))
        safe_d = np.where(parallel, 1.0, d)
        t1 = (lo - a) / safe_d
        t2 = (hi - a) / safe_d
        t_near = np.where(parallel, -np.inf, np.minimum(t1, t2))
        t_far = np.where(parallel, np.inf, np.maximum(t1, t2))
        t_lo = np.maximum(t_lo, t_near)
        t_hi = np.minimum(t_hi, t_far)
    return ok & (t_lo <= t_hi)


def segment_box_distance(A, B, box):
    """Distance from segments A-B to an axis-aligned box (0 if they touch).

    If the segment misses the box, the closest pair of points always involves
    either a segment endpoint or a box corner, so the minimum over those six
    candidate distances is exact.

    `box` is (xmin, ymin, xmax, ymax). Each entry may be a scalar (one box)
    or an array shaped (K, 1, ..., 1) for K boxes at once, in which case the
    result gains a leading axis of length K.
    """
    xmin, ymin, xmax, ymax = box
    d = np.minimum(_point_box_distance(A, box), _point_box_distance(B, box))
    for cx, cy in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)):
        corner = np.stack(np.broadcast_arrays(np.asarray(cx, float), np.asarray(cy, float)), axis=-1)
        d = np.minimum(d, point_segment_distance(corner, A, B))
    return np.where(_segment_hits_box(A, B, box), 0.0, d)


def _orient(a, b, c):
    return ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))


def segment_segment_distance(P1, P2, Q1, Q2):
    """Distance between segments P1-P2 and Q1-Q2 (0 if they cross)."""
    o1, o2 = _orient(P1, P2, Q1), _orient(P1, P2, Q2)
    o3, o4 = _orient(Q1, Q2, P1), _orient(Q1, Q2, P2)
    crossing = (o1 * o2 < 0) & (o3 * o4 < 0)
    d = np.minimum.reduce([
        point_segment_distance(P1, Q1, Q2), point_segment_distance(P2, Q1, Q2),
        point_segment_distance(Q1, P1, P2), point_segment_distance(Q2, P1, P2),
    ])
    return np.where(crossing, 0.0, d)
