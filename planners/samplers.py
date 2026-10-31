"""
Sampling strategies for RRT and RRT*.

Step 1 of both sampling planners is "draw a random configuration". By default
that draw is uniform over the workspace, which is what the original papers
use. This module turns that step into a swappable object so that a *biased*
sampling distribution can be plugged in without touching the planner code.

Every sampler has two methods:

    prepare(env, start, goal)   called once at the start of plan(), INSIDE
                                the planner's timer -- so any set-up cost
                                (e.g. running a neural network) is charged to
                                the planner's computation time.
    sample(env, rng)            called once per iteration; returns (x, y).

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
