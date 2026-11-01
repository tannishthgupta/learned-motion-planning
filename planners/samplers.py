"""
Sampling strategies for RRT and RRT*.

Step 1 of both sampling planners is "draw a random configuration". By default
that draw is uniform over the workspace, which is what the original papers
use. This module turns that step into a swappable object so that a *biased*
sampling distribution can be plugged in without touching the planner code.

Every sampler has three methods:

    prepare(env, start, goal)   called once at the start of plan(), INSIDE
                                the planner's timer -- so any set-up cost
                                (e.g. running a neural network) is charged to
                                the planner's computation time.
    sample(env, rng)            called once per iteration; returns (x, y).
    notify_best_cost(cost)      called by RRT* whenever its best path gets
                                shorter. Only InformedSampler acts on it.

Why a mixture keeps the theory intact
-------------------------------------
`HeatmapSampler` draws from a cell-based heatmap with probability `mix` and
from the uniform distribution otherwise. Because the uniform component keeps
a non-zero density everywhere, every region of free space is still sampled
infinitely often, so RRT keeps probabilistic completeness and RRT* keeps
asymptotic optimality -- whatever the heatmap says. A bad heatmap can only
slow the planner down, never make it miss a solution that exists. This is the
same argument used by Ichter et al. (ICRA 2018) and Neural RRT* (Wang et al.,
IEEE T-ASE 2020).

This file is deliberately free of any ML dependency: the heatmap is just a
NumPy array. The learned network that *produces* a heatmap lives in the
separate `learning/` package.
"""

import bisect
import math

import numpy as np


class UniformSampler:
    """Uniform sampling over the workspace -- the textbook RRT / RRT* step.

    Consumes the RNG exactly as the planners did before samplers existed, so
    every pre-existing result is reproduced bit-for-bit.
    """

    name = "uniform"

    def prepare(self, env, start, goal):
        pass

    def sample(self, env, rng):
        return env.sample(rng)

    def notify_best_cost(self, cost):
        pass

    def __repr__(self):
        return "UniformSampler()"


class HeatmapSampler:
    """Mixture of a cell-based heatmap and the uniform distribution.

    Parameters
    ----------
    heatmap : array (ny, nx) or None
        Non-negative weights over a regular grid covering the workspace; row
        index is y, column index is x. Need not be normalised. May be None if
        a subclass fills it in during prepare().
    mix : float in [0, 1]
        Probability of drawing from the heatmap rather than uniformly.
    """

    name = "heatmap"

    def __init__(self, heatmap=None, mix=0.5):
        if not 0.0 <= mix <= 1.0:
            raise ValueError("mix must be in [0, 1]")
        self.mix = float(mix)
        self._cdf = None
        if heatmap is not None:
            self.set_heatmap(heatmap)

    def set_heatmap(self, heatmap):
        """Build the cumulative table used to pick a cell in O(log n)."""
        w = np.asarray(heatmap, dtype=float)
        if w.ndim != 2:
            raise ValueError("heatmap must be 2-D (ny, nx)")
        w = np.clip(w, 0.0, None)
        total = w.sum()
        self.heatmap = w
        self.ny, self.nx = w.shape
        if total <= 0.0:
            # An all-zero heatmap carries no information: fall back to uniform.
            self._cdf = None
            return
        # A plain Python list makes bisect fast and keeps sample() NumPy-free.
        self._cdf = (np.cumsum(w.ravel()) / total).tolist()

    def prepare(self, env, start, goal):
        pass

    def notify_best_cost(self, cost):
        pass

    def sample(self, env, rng):
        if self._cdf is None or rng.random() >= self.mix:
            return env.sample(rng)

        # 1. pick a cell with probability proportional to its weight
        k = bisect.bisect_left(self._cdf, rng.random())
        k = min(k, len(self._cdf) - 1)
        iy, ix = divmod(k, self.nx)

        # 2. pick a point uniformly inside that cell
        cw = env.width / self.nx
        ch = env.height / self.ny
        x = (ix + rng.random()) * cw
        y = (iy + rng.random()) * ch

        # Keep the whole robot disc inside the workspace, like env.sample().
        r = env.robot_radius
        x = min(max(x, r), env.width - r)
        y = min(max(y, r), env.height - r)
        return (x, y)

    def __repr__(self):
        return "%s(mix=%g)" % (type(self).__name__, self.mix)


class InformedSampler:
    """Informed RRT*: once a path exists, only sample where a better one could pass.

    Reference: Gammell, Srinivasa & Barfoot, "Informed RRT*: Optimal
    Sampling-based Path Planning Focused via Direct Sampling of an Admissible
    Ellipsoidal Heuristic", IROS 2014.

    The idea in one line: if the best path so far has length c_best, then any
    point x on a *shorter* path must satisfy

        |x - start| + |x - goal|  <=  c_best

    which is exactly the inside of an ellipse with the start and goal as its
    two foci (the "string and two pins" construction). Points outside it
    cannot improve the solution, so sampling them is wasted work. Every time
    RRT* finds a shorter path the ellipse shrinks.

    Before the first solution there is no c_best, so this sampler simply
    defers to `base` -- uniform by default, which makes it behave exactly like
    plain RRT* until then.

    Parameters
    ----------
    base : sampler or None
        What to sample from before the first solution, and (when given) the
        proposal distribution afterwards. None = uniform, which gives the
        textbook Informed RRT*. A LearnedSampler here gives "learned +
        informed": samples come from the network's heatmap and are kept only
        if they fall inside the ellipse.
    max_rejections : int
        How many base samples to try before falling back to sampling the
        ellipse directly (only used when `base` is given).
    """

    name = "informed"

    def __init__(self, base=None, max_rejections=20):
        self.use_base_proposal = base is not None
        self.base = base if base is not None else UniformSampler()
        self.max_rejections = int(max_rejections)
        self.c_best = math.inf

    def prepare(self, env, start, goal):
        self.base.prepare(env, start, goal)
        self.c_best = math.inf
        self.dim = len(start)
        if self.dim != 2:
            self._prepare_nd(start, goal)
            return
        self.start = (float(start[0]), float(start[1]))
        self.goal = (float(goal[0]), float(goal[1]))
        dx = self.goal[0] - self.start[0]
        dy = self.goal[1] - self.start[1]
        # c_min: the straight-line distance -- no path can be shorter.
        self.c_min = math.hypot(dx, dy)
        self.centre = ((self.start[0] + self.goal[0]) / 2.0,
                       (self.start[1] + self.goal[1]) / 2.0)
        # Rotation that turns the ellipse's own x-axis onto the start->goal line.
        self.cos_a = dx / self.c_min
        self.sin_a = dy / self.c_min

    @property
    def last_prepare_time(self):
        """Set-up time of the wrapped sampler (e.g. network inference)."""
        return getattr(self.base, "last_prepare_time", 0.0)

    def notify_best_cost(self, cost):
        """Called by RRT* whenever its best path gets shorter."""
        self.c_best = min(self.c_best, float(cost))
        self.base.notify_best_cost(cost)

    def in_ellipse(self, p):
        return (math.dist(p, self.start) + math.dist(p, self.goal)) <= self.c_best

    def sample(self, env, rng):
        if not math.isfinite(self.c_best):
            return self.base.sample(env, rng)          # no path yet

        if self.use_base_proposal:
            for _ in range(self.max_rejections):
                p = self.base.sample(env, rng)
                if self.in_ellipse(p):
                    return p
        return self._sample_ellipse(env, rng)

    # ------------------------------------------------------------------
    # N dimensions (the robot arm): the ellipse becomes an ellipsoid
    # ------------------------------------------------------------------

    def _prepare_nd(self, start, goal):
        """Set up the rotation that maps the unit ball onto the ellipsoid.

        In 2-D a single angle is enough. In N-D the paper builds a rotation
        matrix C whose first column points from start to goal, using an SVD
        (Gammell et al. 2014, Alg. 2). The other N-1 axes can point anywhere
        perpendicular, since the ellipsoid is round across the start-goal line.
        """
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.c_min = float(np.linalg.norm(self.goal - self.start))
        self.centre = (self.start + self.goal) / 2.0
        a1 = (self.goal - self.start) / self.c_min
        e1 = np.zeros(self.dim)
        e1[0] = 1.0
        u, _, vt = np.linalg.svd(np.outer(a1, e1))
        # The middle matrix makes C a proper rotation (determinant +1).
        mid = np.ones(self.dim)
        mid[-1] = np.linalg.det(u) * np.linalg.det(vt)
        self.rotation = u @ np.diag(mid) @ vt

    def _sample_ellipsoid_nd(self, env, rng, max_tries=100):
        """Uniform sample from the N-D ellipsoid, as in _sample_ellipse."""
        d = self.dim
        r1 = self.c_best / 2.0
        r2 = math.sqrt(max(self.c_best ** 2 - self.c_min ** 2, 0.0)) / 2.0
        radii = np.full(d, r2)
        radii[0] = r1
        for _ in range(max_tries):
            # Uniform in the unit ball: a random direction (normalised
            # Gaussian) times a radius u^(1/d), which spreads points evenly
            # by volume -- the N-D version of the sqrt used in 2-D.
            g = np.array([rng.gauss(0.0, 1.0) for _ in range(d)])
            ball = g / np.linalg.norm(g) * rng.random() ** (1.0 / d)
            p = self.rotation @ (radii * ball) + self.centre
            p = tuple(float(v) for v in p)
            if env.in_bounds(p):
                return p
        return env.sample(rng)

    def _sample_ellipse(self, env, rng, max_tries=100):
        """Uniform sample from the ellipse (clipped to the workspace).

        Direct sampling, as in the paper: draw uniformly from the unit disc,
        stretch it into the ellipse, rotate it onto the start-goal line, and
        move it to the midpoint. The semi-axes are

            r1 = c_best / 2                          (along start -> goal)
            r2 = sqrt(c_best^2 - c_min^2) / 2        (across)
        """
        if self.dim != 2:
            return self._sample_ellipsoid_nd(env, rng, max_tries)
        r1 = self.c_best / 2.0
        r2 = math.sqrt(max(self.c_best ** 2 - self.c_min ** 2, 0.0)) / 2.0
        for _ in range(max_tries):
            # Uniform point in the unit disc. The sqrt spreads points evenly
            # by area; without it they would bunch up at the centre.
            rho = math.sqrt(rng.random())
            theta = 2.0 * math.pi * rng.random()
            ux = rho * math.cos(theta) * r1
            uy = rho * math.sin(theta) * r2
            x = self.centre[0] + self.cos_a * ux - self.sin_a * uy
            y = self.centre[1] + self.sin_a * ux + self.cos_a * uy
            # Part of the ellipse may stick out of the world. Reject those
            # points rather than clamping them, which would pile samples up
            # along the walls.
            if env.in_bounds((x, y)):
                return (x, y)
        return env.sample(rng)

    def __repr__(self):
        return "InformedSampler(base=%r)" % (self.base,)
