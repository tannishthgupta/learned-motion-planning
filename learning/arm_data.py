"""
Training data for the arm's learned sampler.

For the 2-D robot the training target was computed exactly (two Dijkstra
searches on a grid). For an arm that is impossible -- a grid over N joint
angles is too big -- so the target comes from planning instead:

    1. draw a random arm world (arm/problems.py, own seed range per split)
    2. run Informed RRT* on it for a long time (REF_ITERS iterations)
    3. keep the best path found, and record K poses spread evenly along it

The network then learns: "for a world like this, with this start and goal,
good paths pass through poses like these".

Honest caveat: a long RRT* run is near-optimal, not optimal. The labels are
"the best path a strong planner found", which is the standard approach in the
literature (Ichter et al. 2018 use the same idea).

Leakage control: the splits use disjoint seed ranges, all different from
anything the evaluation draws.

    train   10_000_000 + i
    val     20_000_000 + i
    test    30_000_000 + i

Usage:
    python -m learning.arm_data --dof 4 --train 700 --val 150
"""

import argparse
import math
import os
import time
from multiprocessing import Pool

import numpy as np

from arm import random_arm_problem
from env.obstacles import CircleObstacle, RectObstacle
from planners import InformedSampler, RRTStarPlanner

SEED_BASE = {"train": 10_000_000, "val": 20_000_000, "test": 30_000_000}

#: Planner settings shared by every arm experiment (data, tuning, evaluation).
ARM_PLANNER = dict(step_size=1.5, goal_tolerance=0.5, goal_bias=0.05)

REF_ITERS = 2500      # length of the reference runs that produce the labels
N_POSES = 24          # poses recorded along each reference path
OCC_GRID = 32         # obstacle image size fed to the network


def occupancy_image(env, size=OCC_GRID):
    """(size, size) bool image of the world, True where an obstacle covers the cell centre.

    Obstacles only, not inflated by the link radius. The arm's base is always
    at the same place, so it does not need to be drawn in.
    """
    xs = (np.arange(size) + 0.5) * env.width / size
    ys = (np.arange(size) + 0.5) * env.height / size
    gx, gy = np.meshgrid(xs, ys)
    occ = np.zeros((size, size), dtype=bool)
    for o in env.obstacles:
        if isinstance(o, CircleObstacle):
            occ |= np.hypot(gx - o.cx, gy - o.cy) <= o.r
        elif isinstance(o, RectObstacle):
            xmin, ymin, xmax, ymax = o.box
            occ |= (gx >= xmin) & (gx <= xmax) & (gy >= ymin) & (gy <= ymax)
    return occ


def resample_path(path, n):
    """n poses spread evenly (by joint-space length) along a path, ends included."""
    P = np.asarray(path, dtype=float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    out = np.empty((n, P.shape[1]))
    for k, t in enumerate(np.linspace(0.0, s[-1], n)):
        i = min(np.searchsorted(s, t, side="right") - 1, len(seg) - 1)
        f = 0.0 if seg[i] == 0 else (t - s[i]) / seg[i]
        out[k] = P[i] + f * (P[i + 1] - P[i])
    return out


def reference_path(env, iters=REF_ITERS, seed=0):
    """Best path a long Informed RRT* run finds, or None."""
    planner = RRTStarPlanner(max_iters=iters, time_budget=600.0, seed=seed,
                             sampler=InformedSampler(), **ARM_PLANNER)
    path, m = planner.plan(env)
    return path, m


def make_record(args):
    split, i, dof = args
    env = random_arm_problem(dof, SEED_BASE[split] + i)
    t0 = time.perf_counter()
    path, m = reference_path(env, seed=i)
    rec = {"occ": occupancy_image(env), "start": np.array(env.start),
           "goal": np.array(env.goal), "solved": bool(m["success"]),
           "cost": m["path_length"] if m["success"] else math.nan,
           "seconds": time.perf_counter() - t0}
    rec["poses"] = (resample_path(path, N_POSES) if m["success"]
                    else np.full((N_POSES, dof), np.nan))
    return rec


def build_split(split, n, dof, out_path, workers=10):
    t0 = time.perf_counter()
    jobs = [(split, i, dof) for i in range(n)]
    recs = []
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap(make_record, jobs, chunksize=1), 1):
            recs.append(r)
            if k % 50 == 0 or k == n:
                el = time.perf_counter() - t0
                print("  %s: %d/%d  (%.0f s, ~%.0f s left)" % (split, k, n, el, el / k * (n - k)),
                      flush=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        occ=np.packbits(np.stack([r["occ"] for r in recs]).reshape(n, -1), axis=1),
        start=np.stack([r["start"] for r in recs]), goal=np.stack([r["goal"] for r in recs]),
        poses=np.stack([r["poses"] for r in recs]), solved=np.array([r["solved"] for r in recs]),
        cost=np.array([r["cost"] for r in recs]), seconds=np.array([r["seconds"] for r in recs]),
        dof=np.array(dof), grid=np.array(OCC_GRID), ref_iters=np.array(REF_ITERS))
    solved = sum(r["solved"] for r in recs)
    print("%s: %d problems, %d solved (%.0f%%), %.0f s -> %s"
          % (split, n, solved, 100.0 * solved / n, time.perf_counter() - t0, out_path))


def load_split(path):
    d = np.load(path)
    g = int(d["grid"])
    n = len(d["start"])
    occ = np.unpackbits(d["occ"], axis=1)[:, :g * g].reshape(n, g, g).astype(bool)
    return {"occ": occ, "start": d["start"], "goal": d["goal"], "poses": d["poses"],
            "solved": d["solved"], "cost": d["cost"], "dof": int(d["dof"])}


def keep_awake():
    """Ask Windows not to sleep while this process runs (no-op elsewhere).

    Long data runs were silently paused by the laptop going to sleep. This is
    the standard request a long-running program makes; it ends with the
    process and does not change any power settings. Closing the lid can still
    suspend the machine, depending on the lid setting.
    """
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dof", type=int, default=4)
    ap.add_argument("--train", type=int, default=700)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join("learning", "data"))
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate splits whose file already exists")
    args = ap.parse_args()
    keep_awake()
    for split, n in (("val", args.val), ("train", args.train)):
        path = os.path.join(args.out, "arm%d_%s.npz" % (args.dof, split))
        if os.path.exists(path) and not args.overwrite:
            print("%s exists, skipping (use --overwrite to regenerate)" % path)
            continue
        build_split(split, n, args.dof, path, args.workers)


if __name__ == "__main__":
    main()
