"""
Generating the training / validation / test problems.

A *problem* is a random world plus a start and a goal. Compared with the main
benchmark (which always plans corner-to-corner) the start and goal here are
randomised, otherwise the network could score well by memorising "the answer
is roughly the main diagonal".

Leakage control
---------------
Each split draws its worlds from its own, non-overlapping seed range, far
away from the seeds `env.generate_suite` uses for the main benchmark:

    train        3_000_000_000 + i
    val          4_000_000_000 + i
    test         5_000_000_000 + i

The hand-designed scenarios (maze, bug trap, narrow gap, ...) are never used
for training at all. They contain long straight walls and C-shapes that the
random rectangle/disc generator never produces, so they measure how the model
copes with map structure it has not seen.

Usage:
    python -m learning.dataset --train 8000 --val 800
"""

import argparse
import math
import os
import random
import time
from multiprocessing import Pool

import numpy as np

from env.generator import random_environment

from .features import GRID, free_mask, to_cell_coords
from .labels import near_optimal_region

SEED_BASE = {"train": 3_000_000_000, "val": 4_000_000_000, "test": 5_000_000_000}

DENSITY_RANGE = (0.05, 0.35)
MIN_SEPARATION = 40.0   # start-goal distance, world units
CORNER_FRACTION = 0.25  # share of problems that use the benchmark's corner layout

_CORNERS = [((5.0, 5.0), (95.0, 95.0)), ((5.0, 95.0), (95.0, 5.0))]


def _random_endpoints(rng, width=100.0, height=100.0, margin=4.0):
    """Start and goal: a corner pair a quarter of the time, otherwise random."""
    if rng.random() < CORNER_FRACTION:
        s, g = _CORNERS[rng.randrange(len(_CORNERS))]
        return (s, g) if rng.random() < 0.5 else (g, s)
    while True:
        s = (rng.uniform(margin, width - margin), rng.uniform(margin, height - margin))
        g = (rng.uniform(margin, width - margin), rng.uniform(margin, height - margin))
        if math.dist(s, g) >= MIN_SEPARATION:
            return s, g


def make_problem(split, i, max_retries=50):
    """Build problem i of a split. Deterministic in (split, i).

    Returns a dict with the world, the grid mask, the label and metadata, or
    raises if no solvable problem turns up (practically never happens).
    """
    base = SEED_BASE[split] + i
    for k in range(max_retries):
        rng = random.Random(base * 100 + k)
        density = rng.uniform(*DENSITY_RANGE)
        start, goal = _random_endpoints(rng)
        env = random_environment(density=density, start=start, goal=goal,
                                 seed=rng.randrange(2 ** 62),
                                 name="%s_%05d" % (split, i))
        free = free_mask(env)
        sc = to_cell_coords(env, start)
        gc = to_cell_coords(env, goal)
        label, d_opt = near_optimal_region(free, (round(sc[0]), round(sc[1])),
                                           (round(gc[0]), round(gc[1])))
        if label is None:
            continue    # unsolvable at grid level -- draw again
        return {"env": env, "free": free, "label": label, "start": start,
                "goal": goal, "density": density, "d_opt": d_opt,
                "seed": base * 100 + k}
    raise RuntimeError("no solvable problem for %s #%d" % (split, i))


def _arrays_for(args):
    split, i = args
    p = make_problem(split, i)
    return (np.packbits(p["free"]), np.packbits(p["label"]),
            p["start"], p["goal"], p["density"], p["seed"])


def build_split(split, n, out_path, workers=None):
    """Generate n problems and save them as one compressed .npz file.

    Masks are bit-packed, so 8000 problems take a few MB on disk. The features
    are rebuilt from (free, start, goal) at training time by the same function
    the planner uses, so nothing is stored twice.
    """
    t0 = time.perf_counter()
    with Pool(workers) as pool:
        out = pool.map(_arrays_for, [(split, i) for i in range(n)], chunksize=16)
    free, label, start, goal, density, seed = zip(*out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path, free=np.stack(free), label=np.stack(label),
        start=np.array(start, dtype=np.float32), goal=np.array(goal, dtype=np.float32),
        density=np.array(density, dtype=np.float32), seed=np.array(seed, dtype=np.int64),
        grid=np.array(GRID))
    print("%s: %d problems in %.1f s -> %s" % (split, n, time.perf_counter() - t0, out_path))
    return out_path


def load_split(path):
    """Load a saved split; masks come back as (n, GRID, GRID) bool arrays."""
    d = np.load(path)
    g = int(d["grid"])
    unpack = lambda a: np.unpackbits(a, axis=1)[:, :g * g].reshape(-1, g, g).astype(bool)
    return {"free": unpack(d["free"]), "label": unpack(d["label"]),
            "start": d["start"], "goal": d["goal"], "density": d["density"],
            "seed": d["seed"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--val", type=int, default=800)
    ap.add_argument("--out", default=os.path.join("learning", "data"))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    build_split("train", args.train, os.path.join(args.out, "train.npz"), args.workers)
    build_split("val", args.val, os.path.join(args.out, "val.npz"), args.workers)


if __name__ == "__main__":
    main()
