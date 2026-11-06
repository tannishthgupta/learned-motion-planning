"""
Evaluate the arm's learned sampler against RRT* and Informed RRT*.

    python -m learning.arm_evaluate --dof 4 --tune   # 1. pick `mix` on VALIDATION worlds
    python -m learning.arm_evaluate --dof 4          # 2. final run on held-out TEST worlds

Same protocol as the 2-D evaluation (learning/evaluate.py), and the same
metric code is reused so the two are measured identically:

  * `mix` is chosen on validation worlds only; the test run reads it from
    results/arm/tuning_arm<dof>.json and changes nothing afterwards.
  * Test worlds come from their own seed range (arm_data.SEED_BASE["test"]),
    never seen in training or tuning.
  * Every method uses the same planner settings (arm_data.ARM_PLANNER) and
    the same seeds; every returned path is re-verified.
  * Primary metric is iterations (wall-clock is noisier with parallel workers).

Some random arm worlds may have no solution at all; they count against every
method equally in the success rate, and are excluded from cost comparisons
because no method has a path there.
"""

import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np

from arm import random_arm_problem
from planners import InformedSampler, RRTStarPlanner, verify_path

from .arm_data import ARM_PLANNER, REF_ITERS, SEED_BASE
from .evaluate import (CHECKPOINTS, _fmt, best_known, cost_at, load_rows,
                       paired_test, summarise, write_histories, write_rows)

OUT_DIR = os.path.join("results", "arm")
METHODS = ["RRT*", "RRT* + informed", "RRT* + learned", "RRT* + learned + informed"]

_MODELS = {}


def _init_worker(model_paths):
    """Load every candidate model once per worker process; one thread each."""
    import torch
    torch.set_num_threads(1)
    from .arm_model import load_model
    for path in model_paths:
        _MODELS[path] = load_model(path)[0]


def make_planner(method, seed, mix, max_iters, model_path=None):
    from .arm_sampler import ArmLearnedSampler
    if method == "reference":
        # The long run used to decide whether a test world is solvable and
        # what its best known cost is -- the same recipe as the training labels.
        return RRTStarPlanner(max_iters=REF_ITERS, seed=seed, sampler=InformedSampler(),
                              name=method, time_budget=600.0, **ARM_PLANNER)
    sampler = (ArmLearnedSampler(_MODELS[model_path], mix=mix, seed=seed)
               if "learned" in method else None)
    if method.endswith("+ informed"):
        sampler = InformedSampler(base=sampler)
    return RRTStarPlanner(max_iters=max_iters, seed=seed, sampler=sampler, name=method,
                          time_budget=600.0, **ARM_PLANNER)


def _run_job(job):
    set_name, dof, problem_seed, method, seed, mix, max_iters, model_path = job
    env = random_arm_problem(dof, problem_seed)
    planner = make_planner(method, seed, mix, max_iters, model_path)
    path, m = planner.plan(env)
    if m["success"]:
        ok, why = verify_path(env, path)
        if not ok:
            raise AssertionError("%s invalid path on %s: %s" % (method, env.name, why))
    return {
        "set": set_name, "problem": env.name, "method": method, "seed": seed,
        "mix": mix if "learned" in method else float("nan"),
        "success": bool(m["success"]),
        "first_iter": m["first_solution_iteration"] if m["success"] else -1,
        "first_time": m["first_solution_time"], "final_cost": m["path_length"],
        "time_sec": m["time_sec"],
        "prep_time": getattr(planner.sampler, "last_prepare_time", 0.0),
        "nodes": m["nodes"],
        "history": [[int(i), float(c)] for i, _, c in planner.cost_history],
        "model": model_path,
    }


def run_jobs(jobs, workers):
    t0 = time.perf_counter()
    rows = []
    model_paths = sorted({j[-1] for j in jobs if j[-1]})
    with Pool(workers, initializer=_init_worker, initargs=(model_paths,)) as pool:
        for k, row in enumerate(pool.imap_unordered(_run_job, jobs, chunksize=1), 1):
            rows.append(row)
            if k % 40 == 0 or k == len(jobs):
                el = time.perf_counter() - t0
                print("  %d/%d runs  %.0f s, ~%.0f s left" % (k, len(jobs), el, el / k * (len(jobs) - k)),
                      flush=True)
    return rows


def tune(args):
    """Choose the training checkpoint AND the mix together, by planning on validation worlds.

    Checkpoints are the files learning/models/arm<dof>_cvae_e<epoch>.pt saved
    by arm_train. The winner is copied to learning/models/arm<dof>_cvae.pt,
    which the test run then uses unchanged.
    """
    import glob
    import re
    import shutil
    pattern = os.path.join("learning", "models", "arm%d_cvae_e*.pt" % args.dof)
    ckpts = sorted(glob.glob(pattern), key=lambda p: int(re.search(r"_e(\d+)\.pt$", p).group(1)))
    if not ckpts:
        raise SystemExit("no checkpoints matching %s -- run learning.arm_train first" % pattern)
    mixes = [0.25, 0.5, 0.9]
    n_val, seeds, iters = 30, 2, 1500
    # Tune only on worlds the long reference run solved: about 30% of random
    # arm worlds have no solution, and they would make every candidate look
    # the same (all fail).
    from .arm_data import load_split
    val = load_split(os.path.join("learning", "data", "arm%d_val.npz" % args.dof))
    worlds = [int(i) for i in np.flatnonzero(val["solved"])[:n_val]]
    ref_cost = {"arm%d_%d" % (args.dof, SEED_BASE["val"] + i): float(val["cost"][i]) for i in worlds}
    jobs = []
    for i in worlds:
        for seed in range(seeds):
            jobs.append(("val", args.dof, SEED_BASE["val"] + i, "RRT*", seed, float("nan"), iters, None))
            for ck in ckpts:
                for mix in mixes:
                    jobs.append(("val", args.dof, SEED_BASE["val"] + i, "RRT* + learned",
                                 seed, mix, iters, ck))
    print("Tuning on %d validation worlds: %d checkpoints x %d mixes, %d runs"
          % (n_val, len(ckpts), len(mixes), len(jobs)))
    rows = run_jobs(jobs, args.workers)

    # Label each candidate by checkpoint epoch and mix, so summarise() treats
    # them as separate methods. Each row records which checkpoint it used.
    for r in rows:
        if r["method"] != "RRT*":
            epoch = re.search(r"_e(\d+)\.pt$", r["model"]).group(1)
            r["method"] = "learned e%s mix=%.2f" % (epoch, r["mix"])
    # Reference cost per world: the best of the long data run and every
    # tuning run, so normalised costs are >= 1.
    ref = dict(ref_cost)
    for (_, prob), c in best_known(rows).items():
        ref[prob] = min(ref.get(prob, math.inf), c)

    ks = (500, 1000, iters)
    groups = {}
    for r in rows:
        groups.setdefault(r["method"], []).append(r)
    table = []
    for method, rs in groups.items():
        succ = {k: success_at(rs, k) for k in ks}
        cost = {k: capped_cost(rs, ref, k) for k in ks}
        table.append((sum(cost.values()) / len(ks), method, succ, cost))
    table.sort()

    best = next(t for t in table if t[1].startswith("learned"))
    epoch, mix = re.match(r"learned e(\d+) mix=([\d.]+)", best[1]).groups()
    chosen_ckpt = pattern.replace("_e*.pt", "_e%s.pt" % epoch)
    final = os.path.join("learning", "models", "arm%d_cvae.pt" % args.dof)
    shutil.copyfile(chosen_ckpt, final)

    lines = ["# Arm (%d joints): checkpoint and mix tuning on validation worlds" % args.dof, "",
             "%d validation worlds that the long reference run solved, x %d seeds, %d iterations. "
             "success@k = share of runs with a path after k iterations. cost@k = mean "
             "normalised cost (1.0 = best known), with a run that has no path yet counted as "
             "%.0f. Score = mean of the cost@k columns; lower is better."
             % (len(worlds), seeds, iters, COST_CAP), "",
             "| candidate | " + " | ".join("success@%d" % k for k in ks) + " | "
             + " | ".join("cost@%d" % k for k in ks) + " | score |",
             "|---|" + "---|" * (2 * len(ks) + 1)]
    for score, method, succ, cost in table:
        lines.append("| %s | %s | %s | %.3f |" % (
            method, " | ".join("%.0f%%" % (100 * succ[k]) for k in ks),
            " | ".join("%.3f" % cost[k] for k in ks), score))
    lines += ["", "**Chosen: epoch %s checkpoint, mix %s** (copied to %s)" % (epoch, mix, final)]
    with open(os.path.join(OUT_DIR, "tuning_arm%d.md" % args.dof), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "tuning_arm%d.json" % args.dof), "w") as f:
        json.dump({"chosen_mix": float(mix), "chosen_epoch": int(epoch),
                   "chosen_checkpoint": chosen_ckpt, "worlds": len(worlds),
                   "scores": {m: s for s, m, _, _ in table}}, f, indent=2)
    print("\n".join(lines))


COST_CAP = 3.0


def success_at(rs, k):
    """Share of runs that have any path within k iterations."""
    return float(np.mean([math.isfinite(cost_at(r["history"], k)) for r in rs]))


def capped_cost(rs, ref, k, cap=COST_CAP):
    """Mean normalised cost at k iterations; no path yet counts as `cap`."""
    vals = []
    for r in rs:
        c = cost_at(r["history"], k) / ref.get(r["problem"], math.inf)
        vals.append(min(c, cap) if math.isfinite(c) else cap)
    return float(np.mean(vals))


def evaluate(args):
    tpath = os.path.join(OUT_DIR, "tuning_arm%d.json" % args.dof)
    mix = args.mix if args.mix is not None else json.load(open(tpath))["chosen_mix"]
    jobs = [("test", args.dof, SEED_BASE["test"] + i, method, seed, mix, args.max_iters,
             args.model if "learned" in method else None)
            for i in range(args.n_test) for seed in range(args.seeds) for method in METHODS]
    # One long reference run per world (seed 100, so it never repeats a test run).
    jobs += [("test", args.dof, SEED_BASE["test"] + i, "reference", 100, float("nan"),
              REF_ITERS, None) for i in range(args.n_test)]
    print("Evaluating %d joints, mix=%.2f: %d runs" % (args.dof, mix, len(jobs)))
    rows = run_jobs(jobs, args.workers)
    for r in rows:
        r.pop("model", None)
    stem = os.path.join(OUT_DIR, "eval_arm%d" % args.dof)
    write_rows(rows, stem + "_raw.csv")
    write_histories(rows, stem + "_histories.json")
    with open(stem + "_meta.json", "w") as f:
        json.dump({"dof": args.dof, "mix": mix, "max_iters": args.max_iters, "seeds": args.seeds,
                   "n_test": args.n_test, "planner": ARM_PLANNER, "model": args.model}, f, indent=2)
    report(rows, args.dof, mix, args.max_iters)


def report(rows, dof, mix, max_iters):
    """Tables for the test run.

    Only worlds with a known solution (the reference run or any method found
    one) are scored; the others are reported as a count. Costs are divided by
    the best cost known for that world (1.0 = best known).
    """
    all_worlds = {r["problem"] for r in rows}
    solvable = {r["problem"] for r in rows if r["success"]}
    rows = [r for r in rows if r["problem"] in solvable]
    ref = {prob: c for (_, prob), c in best_known(rows).items()}
    ks = sorted({250, 500, 1000, max_iters})
    by_method = {m: [r for r in rows if r["method"] == m] for m in METHODS}

    lines = ["# Arm (%d joints): learned sampling, test results" % dof, "",
             "Generated by `python -m learning.arm_evaluate` -- do not edit by hand.", "",
             "%d held-out test worlds; %d have a known solution (found by a %d-iteration "
             "reference run or by any method) and are scored below. %d iterations per run, "
             "planner settings %s, learned mix = %.2f (chosen on validation worlds)."
             % (len(all_worlds), len(solvable), REF_ITERS, max_iters, ARM_PLANNER, mix), "",
             "**success@k** = share of runs with a path after k iterations. **cost@k** = mean "
             "normalised cost after k iterations (1.0 = best known), counting a run with no "
             "path yet as %.0f, so it rewards both finding a path and finding a short one." % COST_CAP,
             "",
             "| method | " + " | ".join("success@%d" % k for k in ks) + " | "
             + " | ".join("cost@%d" % k for k in ks) + " | first path (median iters) | sampler set-up (ms) |",
             "|---|" + "---|" * (2 * len(ks) + 2)]
    table = {}
    for m, rs in by_method.items():
        succ = {k: success_at(rs, k) for k in ks}
        cost = {k: capped_cost(rs, ref, k) for k in ks}
        firsts = [r["first_iter"] for r in rs if r["success"]]
        prep = float(np.median([r["prep_time"] for r in rs])) * 1000
        table[m] = {"success": succ, "cost": cost,
                    "first_iter_median": float(np.median(firsts)) if firsts else None,
                    "prep_ms": prep}
        lines.append("| %s | %s | %s | %s | %.1f |" % (
            m, " | ".join("%.0f%%" % (100 * succ[k]) for k in ks),
            " | ".join("%.3f" % cost[k] for k in ks),
            "%.0f" % np.median(firsts) if firsts else "-", prep))

    # Path quality where every method solved every run: no failures mixed in.
    common = [p for p in solvable if all(r["success"] for r in rows if r["problem"] == p)]
    lines += ["", "Final path cost on the %d worlds where every method solved every run "
              "(median normalised cost at %d iterations):" % (len(common), max_iters), ""]
    for m, rs in by_method.items():
        vals = [r["final_cost"] / ref[r["problem"]] for r in rs if r["problem"] in common]
        lines.append("- %s: %s" % (m, "%.3f" % np.median(vals) if vals else "-"))
        table[m]["common_median_cost"] = float(np.median(vals)) if vals else None

    lines += ["", "## Paired tests (one pair per world, median over seeds; Wilcoxon, two-sided)", "",
              "| comparison | k | worlds | A better | B better | median A/B | p |",
              "|---|---|---|---|---|---|---|"]
    tests = []
    for k in sorted({500, 1000, max_iters}):
        for a, b in [("RRT* + learned", "RRT*"), ("RRT* + informed", "RRT*"),
                     ("RRT* + learned", "RRT* + informed"),
                     ("RRT* + learned + informed", "RRT* + informed")]:
            t = paired_test(rows, "test", a, b, k)
            if t:
                tests.append(t)
                lines.append("| %s vs %s | %d | %d | %.0f%% | %.0f%% | %.3f | %.2g |" % (
                    a, b, k, t["n_problems"], 100 * t["a_better"], 100 * t["b_better"],
                    t["median_ratio"], t["p_value"]))
    path = os.path.join(OUT_DIR, "eval_arm%d_summary.md" % dof)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "eval_arm%d_summary.json" % dof), "w") as f:
        json.dump({"worlds": len(all_worlds), "solvable": len(solvable), "common": len(common),
                   "methods": table, "tests": tests}, f, indent=2, default=lambda x: None)
    print("\n".join(lines))
    print("Wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dof", type=int, default=4)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--mix", type=float, default=None)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    args.model = args.model or os.path.join("learning", "models", "arm%d_cvae.pt" % args.dof)
    os.makedirs(OUT_DIR, exist_ok=True)
    from .arm_data import keep_awake
    keep_awake()
    if args.tune:
        tune(args)
    elif args.from_cache:
        stem = os.path.join(OUT_DIR, "eval_arm%d" % args.dof)
        meta = json.load(open(stem + "_meta.json"))
        report(load_rows(stem + "_raw.csv", stem + "_histories.json"),
               args.dof, meta["mix"], meta["max_iters"])
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
