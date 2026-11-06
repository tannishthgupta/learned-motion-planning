"""
Plug the arm pose generator into RRT / RRT* (same interface as
planners/samplers.py).

At the start of planning (prepare, inside the planner's timer) the network
generates a batch of poses for this problem in one forward pass. Each call
to sample() then returns one of those poses with probability `mix`, or a
uniform random pose otherwise. The uniform share keeps RRT*'s completeness
and optimality guarantees whatever the network says -- the same argument as
for the 2-D heatmap sampler.

Wrap it in InformedSampler for "learned + informed": once a path exists,
network poses outside the Informed RRT* ellipsoid are rejected.
"""

import time

import numpy as np
import torch

from .arm_data import occupancy_image
from .arm_model import load_model


class ArmLearnedSampler:
    """
    Parameters
    ----------
    model : ArmCVAE or path to a saved one
    mix : probability of using a network pose instead of a uniform one
    batch : poses generated per forward pass (more are made if they run out)
    seed : seeds the network's noise, so runs are reproducible
    """

    name = "learned"

    def __init__(self, model, mix=0.5, batch=2048, seed=0):
        if isinstance(model, str):
            model, _ = load_model(model)
        self.model = model.eval()
        self.mix = float(mix)
        self.batch = int(batch)
        self.seed = int(seed)
        self.last_prepare_time = 0.0

    def prepare(self, env, start, goal):
        t0 = time.perf_counter()
        self._occ = torch.from_numpy(occupancy_image(env, self.model.grid).astype(np.float32))
        self._start = torch.tensor(start, dtype=torch.float32)
        self._goal = torch.tensor(goal, dtype=torch.float32)
        self._gen = torch.Generator().manual_seed(self.seed)
        self._refill()
        self.last_prepare_time = time.perf_counter() - t0

    def _refill(self):
        q = self.model.generate(self._occ, self._start, self._goal, self.batch,
                                generator=self._gen).numpy().astype(float)
        self._buffer = [tuple(row) for row in q]
        self._next = 0

    def notify_best_cost(self, cost):
        pass

    def sample(self, env, rng):
        if rng.random() >= self.mix:
            return env.sample(rng)
        if self._next == len(self._buffer):
            self._refill()
        q = self._buffer[self._next]
        self._next += 1
        return q

    def __repr__(self):
        return "ArmLearnedSampler(mix=%g)" % self.mix
