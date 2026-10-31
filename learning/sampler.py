"""
Samplers that plug into RRT / RRT* (see planners/samplers.py for the interface).

    LearnedSampler   heatmap = the network's prediction
    DijkstraSampler  heatmap = the exact near-optimal region from labels.py

Both are a `HeatmapSampler`, so they share the same mixture rule (heatmap with
probability `mix`, uniform otherwise) and the same theoretical guarantees.
They differ only in where the heatmap comes from.

Why DijkstraSampler exists
--------------------------
On a 2-D 64 x 64 grid the exact target the network is trained to imitate can
be computed directly in a few milliseconds. DijkstraSampler does that, which
gives the evaluation an honest reference point:

  * learned vs uniform     -> does learning help at all?
  * learned vs Dijkstra    -> how much of the achievable gain does the
                              network recover?

In 2-D you would simply use Dijkstra. The learned sampler matters for
high-dimensional problems (robot arms, 6+ joints) where a grid over the whole
space is impossible to build; 2-D is used here as a controlled testbed where
the ideal answer is known, so the gap can be measured exactly.

Both samplers do all their set-up inside prepare(), which the planners call
inside their timer, so the cost of inference / Dijkstra is part of the
reported planning time.
"""

import time

import numpy as np
import torch

from planners.samplers import HeatmapSampler

from .features import problem_features, to_cell_coords
from .labels import near_optimal_region
from .model import load_model


class LearnedSampler(HeatmapSampler):
    """Sample from the U-Net's predicted near-optimal region.

    Parameters
    ----------
    model : SamplingUNet or str
        A loaded model, or a path to a saved one.
    mix : float
        Probability of sampling from the heatmap instead of uniformly.
    """

    name = "learned"

    def __init__(self, model, mix=0.5):
        super().__init__(heatmap=None, mix=mix)
        if isinstance(model, str):
            model, _ = load_model(model)
        self.model = model.eval()
        self.last_prepare_time = 0.0
        self.last_probabilities = None

    @torch.no_grad()
    def prepare(self, env, start, goal):
        t0 = time.perf_counter()
        x, free = problem_features(env, start, goal)
        logits = self.model(torch.from_numpy(x)[None])[0].numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))
        # Never waste a sample on a cell the robot cannot occupy.
        prob = prob * free
        self.last_probabilities = prob
        self.set_heatmap(prob)
        self.last_prepare_time = time.perf_counter() - t0


class DijkstraSampler(HeatmapSampler):
    """Sample from the exact near-optimal region (the network's training target)."""

    name = "dijkstra"

    def __init__(self, mix=0.5, eps=0.02):
        super().__init__(heatmap=None, mix=mix)
        self.eps = eps
        self.last_prepare_time = 0.0
        self.last_probabilities = None

    def prepare(self, env, start, goal):
        t0 = time.perf_counter()
        _, free = problem_features(env, start, goal)
        sc = to_cell_coords(env, start)
        gc = to_cell_coords(env, goal)
        label, _ = near_optimal_region(free, (round(sc[0]), round(sc[1])),
                                       (round(gc[0]), round(gc[1])), eps=self.eps)
        heat = (label.astype(float) if label is not None
                else np.zeros_like(free, dtype=float))
        self.last_probabilities = heat
        self.set_heatmap(heat)   # all-zero -> falls back to uniform
        self.last_prepare_time = time.perf_counter() - t0
