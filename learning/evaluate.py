"""
Evaluate the learned sampler against uniform and exact-Dijkstra sampling.

    python -m learning.evaluate --tune      # 1. pick `mix` on VALIDATION maps
    python -m learning.evaluate             # 2. final run on the TEST sets
    python -m learning.evaluate --quick     #    small smoke test

Step 1 is the only place a hyper-parameter is chosen; step 2 reads the chosen
value from results/learned/tuning.json and never feeds anything back. That
keeps the test numbers an honest estimate of unseen performance.

Test sets
---------
    test        100 held-out random problems, random start/goal
                (same distribution as training, different worlds)
    benchmark   the 30 worlds of the main project benchmark (corner-to-corner)
    scenarios   the 5 hand-designed worlds -- map structure never seen in
                training (long walls, a C-shaped trap, a single doorway)

Methods (identical step size, goal bias, tolerance and iteration cap):
    RRT*, RRT* + informed (Informed RRT*, Gammell et al. 2014),
    RRT* + learned, RRT* + learned + informed, RRT* + Dijkstra,
    RRT, RRT + learned

What is measured
----------------
RRT* is an anytime planner, so a single final number hides most of the story.
For each run we keep the whole best-cost-vs-iteration curve and report:

  * cost after k iterations, k in CHECKPOINTS, divided by the best cost any
    method found on that problem ("normalised cost", 1.00 = best known)
  * iterations until the cost is within 5% of that best known cost
  * iterations / seconds to the first solution, success rate
  * sampler set-up time (network inference or Dijkstra), already included in
    the planner's wall-clock time

Iteration counts are the primary metric: runs execute in parallel worker
processes, which makes wall-clock times noisier than they would be in
isolation (but equally so for every method).
"""

import argparse
import csv
import json
import math
import os
import statistics
import time
from multiprocessing import Pool

import numpy as np

from env import generate_suite, get_scenario, SCENARIOS
from planners import InformedSampler, RRTPlanner, RRTStarPlanner, verify_path

from .dataset import make_problem

OUT_DIR = os.path.join("results", "learned")
MODEL_PATH = os.path.join("learning", "models", "sampler_unet.pt")

CHECKPOINTS = [100, 250, 500, 1000, 2000, 5000]
NEAR_OPT = 1.05   # "within 5% of best known"

COMMON = dict(step_size=5.0, goal_bias=0.05, goal_tolerance=5.0, time_budget=10.0)

RRTSTAR_METHODS = ["RRT*", "RRT* + informed", "RRT* + learned",
                   "RRT* + learned + informed", "RRT* + Dijkstra"]
RRT_METHODS = ["RRT", "RRT + learned"]

# ----------------------------------------------------------------------------
# Worker side
# ----------------------------------------------------------------------------

_MODEL = None


def _init_worker(model_path):
    """Load the network once per worker process; one thread each."""
    global _MODEL
    import torch
    torch.set_num_threads(1)
    if model_path:
        from .model import load_model
        _MODEL, _ = load_model(model_path)


def make_planner(method, seed, mix, max_iters):
    """Build a planner from its name, e.g. "RRT* + learned + informed"."""
    from .sampler import DijkstraSampler, LearnedSampler
    sampler = None
    if "+ learned" in method:
        sampler = LearnedSampler(_MODEL, mix=mix)
    elif "+ Dijkstra" in method:
        sampler = DijkstraSampler(mix=mix)
    if method.endswith("+ informed"):
        # Wraps whatever was chosen above (None = uniform = textbook Informed RRT*).
        sampler = InformedSampler(base=sampler)
    cls = RRTStarPlanner if method.startswith("RRT*") else RRTPlanner
    return cls(max_iters=max_iters, seed=seed, sampler=sampler, name=method, **COMMON)


def _run_job(job):
    set_name, env, method, seed, mix, max_iters = job
    planner = make_planner(method, seed, mix, max_iters)
    path, m = planner.plan(env)
    if m["success"]:
        ok, why = verify_path(env, path)
        if not ok:
            raise AssertionError("%s invalid path on %s: %s" % (method, env.name, why))
    if isinstance(planner, RRTStarPlanner):
        history = [[int(i), float(c)] for i, _, c in planner.cost_history]
    else:
        history = ([[m["first_solution_iteration"], m["path_length"]]]
                   if m["success"] else [])
    return {
        "set": set_name, "problem": env.name, "method": method, "seed": seed,
        "mix": mix if ("learned" in method or "Dijkstra" in method) else float("nan"),
        "success": bool(m["success"]),
        "first_iter": m["first_solution_iteration"] if m["success"] else -1,
        "first_time": m["first_solution_time"],
        "final_cost": m["path_length"],
        "time_sec": m["time_sec"],
        "prep_time": getattr(planner.sampler, "last_prepare_time", 0.0),
        "nodes": m["nodes"],
        "history": history,
    }


# ----------------------------------------------------------------------------
# Problem sets
# ----------------------------------------------------------------------------

def problem_sets(n_test=100, quick=False):
    sets = {}
    sets["test"] = [make_problem("test", i)["env"] for i in range(10 if quick else n_test)]
    suite = generate_suite(n_per_density=2 if quick else 10, seed=0)
    sets["benchmark"] = suite
    sets["scenarios"] = [get_scenario(name) for name in SCENARIOS]
    return sets


def run_jobs(jobs, model_path, workers):
    t0 = time.perf_counter()
    rows = []
    with Pool(workers, initializer=_init_worker, initargs=(model_path,)) as pool:
        for k, row in enumerate(pool.imap_unordered(_run_job, jobs, chunksize=2), 1):
            rows.append(row)
            if k % 100 == 0 or k == len(jobs):
                el = time.perf_counter() - t0
                print("  %5d/%d runs  %.0f s elapsed, ~%.0f s left"
                      % (k, len(jobs), el, el / k * (len(jobs) - k)), flush=True)
    return rows


# ----------------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------------

def cost_at(history, k):
    """Best cost found within the first k iterations (inf if none yet)."""
    best = math.inf
    for it, c in history:
        if it <= k:
            best = min(best, c)
        else:
            break
    return best


def iters_to(history, target):
    """First iteration at which the cost is <= target (inf if never)."""
    for it, c in history:
        if c <= target:
            return it
    return math.inf


def best_known(rows):
    """Per-problem reference: the lowest final cost any method reached."""
    ref = {}
    for r in rows:
        if r["success"]:
            key = (r["set"], r["problem"])
            ref[key] = min(ref.get(key, math.inf), r["final_cost"])
    return ref


def _median(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(xs) if xs else float("nan")


def _fmt(x, nd=3):
    if isinstance(x, float) and math.isinf(x):
        return "not reached"
    if isinstance(x, float) and math.isnan(x):
        return "-"
    return ("%%.%df" % nd) % x


def summarise(rows, checkpoints, max_iters):
    """One summary row per (set, method)."""
    ref = best_known(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r["set"], r["method"]), []).append(r)

    out = []
    for (set_name, method), rs in sorted(groups.items()):
        row = {"set": set_name, "method": method, "runs": len(rs),
               "success": sum(r["success"] for r in rs) / len(rs),
               "first_iter": _median([r["first_iter"] if r["success"] else math.inf
                                      for r in rs]),
               "first_time_ms": 1000 * _median([r["first_time"] if r["success"] else math.inf
                                                for r in rs]),
               "prep_ms": 1000 * _median([r["prep_time"] for r in rs])}
        for k in checkpoints:
            if k > max_iters:
                continue
            row["cost@%d" % k] = _median([cost_at(r["history"], k) / ref[(r["set"], r["problem"])]
                                          if (r["set"], r["problem"]) in ref else math.inf
                                          for r in rs])
        row["iters_to_5pct"] = _median([iters_to(r["history"], NEAR_OPT * ref[(r["set"], r["problem"])])
                                        if (r["set"], r["problem"]) in ref else math.inf
                                        for r in rs])
        out.append(row)
    return out


def paired_test(rows, set_name, a, b, k):
    """Paired comparison of methods a vs b at k iterations, one pair per problem.

    For each problem, take each method's median normalised cost over seeds.
    Reports the fraction of problems where `a` is strictly better, the median
    ratio a/b, and a two-sided Wilcoxon signed-rank p-value.
    """
    from scipy.stats import wilcoxon
    ref = best_known(rows)
    per = {}
    for r in rows:
        if r["set"] != set_name or r["method"] not in (a, b):
            continue
        key = (r["set"], r["problem"])
        if key not in ref:
            continue
        per.setdefault(r["problem"], {}).setdefault(r["method"], []).append(
            cost_at(r["history"], k) / ref[key])
    va, vb = [], []
    for prob, d in per.items():
        if a in d and b in d:
            ma, mb = statistics.median(d[a]), statistics.median(d[b])
            if math.isinf(ma) and math.isinf(mb):
                continue
            va.append(ma)
            vb.append(mb)
    if len(va) < 5:
        return None
    big = 10.0   # an unsolved run counts as "10x the best cost" for ranking
    fa = np.minimum(np.array(va), big)
    fb = np.minimum(np.array(vb), big)
    wins = float(np.mean(fa < fb - 1e-9))
    losses = float(np.mean(fa > fb + 1e-9))
    diff = fa - fb
    p = float(wilcoxon(fa, fb).pvalue) if np.any(np.abs(diff) > 1e-12) else 1.0
    return {"set": set_name, "a": a, "b": b, "k": k, "n_problems": len(va),
            "a_better": wins, "b_better": losses,
            "median_ratio": float(np.median(fa / fb)), "p_value": p}


def write_rows(rows, path):
    fields = [k for k in rows[0] if k != "history"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def write_histories(rows, path):
    with open(path, "w") as f:
        json.dump([{k: r[k] for k in ("set", "problem", "method", "seed", "history")}
                   for r in rows], f)


def load_rows(raw_csv, hist_json):
    hist = {(h["set"], h["problem"], h["method"], int(h["seed"])): h["history"]
            for h in json.load(open(hist_json))}
    rows = []
    with open(raw_csv, newline="") as f:
        for r in csv.DictReader(f):
            r["seed"] = int(r["seed"])
            r["success"] = r["success"] == "True"
            for k in ("mix", "first_time", "final_cost", "time_sec", "prep_time"):
                r[k] = float(r[k])
            r["first_iter"] = int(r["first_iter"])
            r["nodes"] = int(r["nodes"])
            r["history"] = hist[(r["set"], r["problem"], r["method"], r["seed"])]
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------
# Tuning (validation only)
# ----------------------------------------------------------------------------

def tune(args):
    mixes = [0.25, 0.5, 0.75, 0.9]
    n_val, seeds, max_iters = (8, 2, 1000) if args.quick else (40, 3, 2000)
    envs = [make_problem("val", i)["env"] for i in range(n_val)]
    jobs = []
    for env in envs:
        for seed in range(seeds):
            jobs.append(("val", env, "RRT*", seed, float("nan"), max_iters))
            for mix in mixes:
                jobs.append(("val", env, "RRT* + learned", seed, mix, max_iters))
    print("Tuning on %d validation problems, %d runs" % (len(envs), len(jobs)))
    rows = run_jobs(jobs, args.model, args.workers)

    # Treat each mix as its own method for summarising.
    for r in rows:
        if r["method"] != "RRT*":
            r["method"] = "RRT* + learned (mix=%.2f)" % r["mix"]
    cps = [k for k in CHECKPOINTS if k <= max_iters]
    summary = summarise(rows, cps, max_iters)

    def score(s):
        # Average normalised cost across checkpoints (lower is better).
        vals = [min(s["cost@%d" % k], 10.0) for k in cps if k >= 250]
        return sum(vals) / len(vals)

    learned = [s for s in summary if "mix=" in s["method"]]
    best = min(learned, key=score)
    chosen = float(best["method"].split("mix=")[1].rstrip(")"))

    os.makedirs(OUT_DIR, exist_ok=True)
    lines = ["# Mix tuning (validation set)", "",
             "%d validation problems x %d seeds, %d iterations. Score = mean "
             "normalised cost over checkpoints >= 250 (lower is better)."
             % (n_val, seeds, max_iters), "",
             "| method | " + " | ".join("cost@%d" % k for k in cps) + " | score |",
             "|---|" + "---|" * (len(cps) + 1)]
    for s in summary:
        lines.append("| %s | %s | %.4f |" % (
            s["method"], " | ".join(_fmt(s["cost@%d" % k]) for k in cps), score(s)))
    lines += ["", "**Chosen mix: %.2f**" % chosen]
    suffix = "_quick" if args.quick else ""
    with open(os.path.join(OUT_DIR, "tuning%s.md" % suffix), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "tuning%s.json" % suffix), "w") as f:
        json.dump({"chosen_mix": chosen, "candidates": mixes, "n_val": n_val,
                   "seeds": seeds, "max_iters": max_iters,
                   "scores": {s["method"]: score(s) for s in summary}}, f, indent=2)
    print("\n".join(lines))


# ----------------------------------------------------------------------------
# Final evaluation
# ----------------------------------------------------------------------------

def evaluate(args):
    suffix = "_quick" if args.quick else ""
    if args.mix is not None:
        mix = args.mix
    else:
        tpath = os.path.join(OUT_DIR, "tuning%s.json" % suffix)
        if not os.path.exists(tpath):
            raise SystemExit("run `python -m learning.evaluate --tune` first (or pass --mix)")
        mix = json.load(open(tpath))["chosen_mix"]

    max_iters = 1000 if args.quick else args.max_iters
    sets = problem_sets(quick=args.quick)
    seeds = {"test": 5, "benchmark": 5, "scenarios": 20}
    if args.quick:
        seeds = {"test": 2, "benchmark": 2, "scenarios": 3}

    jobs = []
    for set_name, envs in sets.items():
        for env in envs:
            for seed in range(seeds[set_name]):
                for method in RRTSTAR_METHODS + RRT_METHODS:
                    jobs.append((set_name, env, method, seed, mix, max_iters))
    print("Evaluating mix=%.2f: %d runs over %s" % (
        mix, len(jobs), ", ".join("%s(%d)" % (k, len(v)) for k, v in sets.items())))
    rows = run_jobs(jobs, args.model, args.workers)

    os.makedirs(OUT_DIR, exist_ok=True)
    raw = os.path.join(OUT_DIR, "eval_raw%s.csv" % suffix)
    hist = os.path.join(OUT_DIR, "eval_histories%s.json" % suffix)
    write_rows(rows, raw)
    write_histories(rows, hist)
    with open(os.path.join(OUT_DIR, "eval_meta%s.json" % suffix), "w") as f:
        json.dump({"mix": mix, "max_iters": max_iters, "seeds": seeds,
                   "common": COMMON, "workers": args.workers,
                   "model": args.model, "sets": {k: [e.name for e in v]
                                                 for k, v in sets.items()}}, f, indent=2)
    report(rows, max_iters, mix, suffix)


def report(rows, max_iters, mix, suffix=""):
    cps = [k for k in CHECKPOINTS if k <= max_iters]
    summary = summarise(rows, cps, max_iters)
    lines = ["# Learned sampling: evaluation results", "",
             "Generated by `python -m learning.evaluate` -- do not edit by hand.", "",
             "mix = %.2f (chosen on the validation set), %d iterations per run. "
             "Normalised cost = best cost so far / best cost any method found on "
             "that problem (1.000 = best known). Values are medians over all runs; "
             "\"not reached\" means the median run had no solution yet." % (mix, max_iters), ""]
    for set_name in ["test", "benchmark", "scenarios"]:
        ss = [s for s in summary if s["set"] == set_name]
        if not ss:
            continue
        lines += ["## %s" % set_name, "",
                  "| method | success | first sol. (iters) | first sol. (ms) | sampler set-up (ms) | "
                  + " | ".join("cost@%d" % k for k in cps) + " | iters to within 5% |",
                  "|---|" + "---|" * (len(cps) + 5)]
        for s in ss:
            lines.append("| %s | %.0f%% | %s | %s | %s | %s | %s |" % (
                s["method"], 100 * s["success"], _fmt(s["first_iter"], 0),
                _fmt(s["first_time_ms"], 1), _fmt(s["prep_ms"], 1),
                " | ".join(_fmt(s["cost@%d" % k]) for k in cps), _fmt(s["iters_to_5pct"], 0)))
        lines.append("")

    lines += ["## Paired significance tests", "",
              "One pair per problem (median over seeds). Wilcoxon signed-rank, two-sided.", "",
              "| set | comparison | k | problems | A better | B better | median A/B | p |",
              "|---|---|---|---|---|---|---|---|"]
    tests = []
    for set_name in ["test", "benchmark"]:
        for k in sorted({k for k in (500, 1000, max_iters) if k <= max_iters}):
            for a, b in [("RRT* + learned", "RRT*"),
                         ("RRT* + informed", "RRT*"),
                         ("RRT* + learned", "RRT* + informed"),
                         ("RRT* + learned + informed", "RRT* + informed"),
                         ("RRT* + learned + informed", "RRT* + learned"),
                         ("RRT* + learned", "RRT* + Dijkstra")]:
                t = paired_test(rows, set_name, a, b, k)
                if t:
                    tests.append(t)
                    lines.append("| %s | %s vs %s | %d | %d | %.0f%% | %.0f%% | %.3f | %.2g |" % (
                        set_name, a, b, k, t["n_problems"], 100 * t["a_better"],
                        100 * t["b_better"], t["median_ratio"], t["p_value"]))
    lines.append("")

    # Per-scenario breakdown: these are few enough to show individually.
    sc = [r for r in rows if r["set"] == "scenarios"]
    if sc:
        ks = sorted({k for k in (1000, max_iters) if k <= max_iters})
        lines += ["## Scenarios, one by one", "",
                  "| scenario | method | success | first sol. (iters) | "
                  + " | ".join("cost@%d" % k for k in ks) + " |",
                  "|---|---|---|---|" + "---|" * len(ks)]
        for prob in sorted({r["problem"] for r in sc}):
            sub = [r for r in sc if r["problem"] == prob]
            for s in summarise(sub, ks, max_iters):
                lines.append("| %s | %s | %.0f%% | %s | %s |" % (
                    prob, s["method"], 100 * s["success"], _fmt(s["first_iter"], 0),
                    " | ".join(_fmt(s["cost@%d" % k]) for k in ks)))
        lines.append("")

    md = os.path.join(OUT_DIR, "eval_summary%s.md" % suffix)
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, "eval_summary%s.json" % suffix), "w") as f:
        json.dump({"summary": summary, "tests": tests}, f, indent=2,
                  default=lambda x: None)
    print("\n".join(lines))
    print("Wrote", md)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", action="store_true", help="choose mix on validation maps")
    ap.add_argument("--from-cache", action="store_true", help="re-analyse saved results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mix", type=float, default=None, help="override the tuned mix")
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    if args.tune:
        tune(args)
    elif args.from_cache:
        suffix = "_quick" if args.quick else ""
        meta = json.load(open(os.path.join(OUT_DIR, "eval_meta%s.json" % suffix)))
        rows = load_rows(os.path.join(OUT_DIR, "eval_raw%s.csv" % suffix),
                         os.path.join(OUT_DIR, "eval_histories%s.json" % suffix))
        report(rows, meta["max_iters"], meta["mix"], suffix)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
