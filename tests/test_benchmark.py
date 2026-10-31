"""
Tests for the benchmark harness, the aggregation, and the charts.

The important ones here are about *aggregation bias*. The report's headline
claim is a comparison of path lengths, and that comparison is only valid if
every planner is scored on the same instances. `test_common_subset_*` builds a
case where the naive average is actively misleading and checks that the
common-solved average is not.

Run with:  python -m unittest discover -s tests -t . -v
"""

import os
import shutil
import tempfile
import unittest

from benchmark import (BenchmarkConfig, common_solved_instances, markdown_tables,
                       read_rows, run_benchmark, summarise, write_rows)
from planners import make_metrics


def _row(planner, env, trial, success, length, time_sec=0.01, density="sparse",
         nodes=100):
    """One synthetic benchmark row."""
    m = make_metrics(planner=planner, environment=env, success=success,
                     path_length=length if success else float("nan"),
                     time_sec=time_sec, nodes=nodes,
                     first_solution_time=time_sec if success else float("nan"),
                     first_solution_length=length if success else float("nan"))
    row = {"density_label": density, "density_target": 0.1, "trial": trial,
           "seed": trial}
    row.update(m)
    return row


class TestCommonSolvedSubset(unittest.TestCase):
    """Quality metrics must be averaged over instances everyone solved."""

    def setUp(self):
        # Two worlds. "easy" is short for everyone. "hard" is long, and RRT
        # fails on it -- so RRT is only ever scored on the easy world.
        self.rows = [
            _row("A*", "easy", 0, True, 100.0),
            _row("RRT", "easy", 0, True, 130.0),
            _row("RRT*", "easy", 0, True, 95.0),

            _row("A*", "hard", 0, True, 300.0),
            _row("RRT", "hard", 0, False, None),
            _row("RRT*", "hard", 0, True, 290.0),
        ]

    def test_identifies_the_intersection(self):
        common = common_solved_instances(self.rows)
        self.assertEqual(common, {("easy", 0)})

    def test_naive_mean_would_flatter_the_failing_planner(self):
        """The bug this machinery exists to prevent.

        Averaged over its own successes RRT scores 130 against A*'s 200, which
        would read as RRT producing *shorter* paths. It does not -- it simply
        never attempted the long world.
        """
        summary = summarise(self.rows)
        by = {s["planner"]: s for s in summary if s["density_label"] == "all"}

        self.assertAlmostEqual(by["RRT"]["length_mean"], 130.0)
        self.assertAlmostEqual(by["A*"]["length_mean"], 200.0)
        self.assertLess(by["RRT"]["length_mean"], by["A*"]["length_mean"])

        # On the common subset the ordering is corrected.
        self.assertAlmostEqual(by["RRT"]["length_mean_common"], 130.0)
        self.assertAlmostEqual(by["A*"]["length_mean_common"], 100.0)
        self.assertGreater(by["RRT"]["length_mean_common"],
                           by["A*"]["length_mean_common"])

    def test_failures_show_up_in_the_success_rate(self):
        """The failure is not hidden -- it is moved to the metric that owns it."""
        summary = summarise(self.rows)
        by = {s["planner"]: s for s in summary if s["density_label"] == "all"}
        self.assertAlmostEqual(by["RRT"]["success_rate"], 50.0)
        self.assertAlmostEqual(by["A*"]["success_rate"], 100.0)

    def test_normalised_length_is_relative_to_the_best_seen(self):
        summary = summarise(self.rows)
        by = {s["planner"]: s for s in summary if s["density_label"] == "all"}
        # On "easy" the best is RRT* at 95, so RRT* normalises to exactly 1.
        self.assertAlmostEqual(by["RRT*"]["length_norm_mean"], 1.0)
        self.assertAlmostEqual(by["RRT"]["length_norm_mean"], 130.0 / 95.0)


class TestSummarise(unittest.TestCase):

    def test_counts_and_rates(self):
        rows = [_row("A*", "e%d" % i, 0, i % 2 == 0, 100.0) for i in range(4)]
        summary = summarise(rows, planners=["A*"])
        row = [s for s in summary if s["density_label"] == "all"][0]
        self.assertEqual(row["n_runs"], 4)
        self.assertEqual(row["n_success"], 2)
        self.assertAlmostEqual(row["success_rate"], 50.0)

    def test_per_density_and_pooled_rows_are_both_present(self):
        rows = ([_row("A*", "a", 0, True, 100.0, density="sparse")] +
                [_row("A*", "b", 0, True, 200.0, density="cluttered")])
        summary = summarise(rows, planners=["A*"])
        labels = {s["density_label"] for s in summary}
        self.assertEqual(labels, {"sparse", "cluttered", "all"})
        pooled = [s for s in summary if s["density_label"] == "all"][0]
        self.assertAlmostEqual(pooled["length_mean"], 150.0)

    def test_markdown_renders_without_error(self):
        rows = [_row(p, "a", 0, True, 100.0) for p in ("A*", "RRT", "RRT*")]
        text = markdown_tables(summarise(rows), rows)
        self.assertIn("Headline comparison", text)
        self.assertIn("RRT*", text)


class TestRunnerIO(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_csv_round_trip_preserves_types(self):
        rows = [_row("A*", "a", 0, True, 123.456, time_sec=0.25),
                _row("RRT", "a", 0, False, None)]
        path = write_rows(rows, os.path.join(self.tmp, "raw.csv"))
        back = read_rows(path)

        self.assertEqual(len(back), 2)
        self.assertIs(back[0]["success"], True)
        self.assertIs(back[1]["success"], False)
        self.assertAlmostEqual(back[0]["path_length"], 123.456)
        self.assertAlmostEqual(back[0]["time_sec"], 0.25)
        self.assertIsInstance(back[0]["nodes"], int)


class TestEndToEndBenchmark(unittest.TestCase):
    """A real (tiny) sweep, so the harness itself is covered."""

    def test_small_sweep_produces_consistent_rows(self):
        config = BenchmarkConfig(densities={"sparse": 0.10}, n_envs_per_density=2,
                                 n_trials=2, time_budget=5.0, max_iters=1500)
        rows, convergence, suite = run_benchmark(config, verbose=False)

        # 2 worlds x 2 trials x 3 planners.
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(suite), 2)
        self.assertTrue(all(r["density_label"] == "sparse" for r in rows))

        # run_benchmark re-verifies every successful path internally, so
        # reaching this point already means none of them were invalid.
        for row in rows:
            if row["success"]:
                self.assertGreater(row["path_length"], 0.0)
                self.assertGreaterEqual(row["time_sec"], 0.0)

        # RRT* must have produced at least one refinement curve.
        self.assertTrue(convergence)
        for curve in convergence:
            costs = [c for _, _, c in curve["history"]]
            for earlier, later in zip(costs, costs[1:]):
                self.assertLessEqual(later, earlier + 1e-9)


class TestCharts(unittest.TestCase):
    """Charts must render from real summary data without raising."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.rows = []
        for density in ("sparse", "cluttered"):
            for i in range(3):
                self.rows.append(_row("A*", "%s%d" % (density, i), 0, True,
                                      140.0 + i, 0.03, density))
                self.rows.append(_row("RRT", "%s%d" % (density, i), 0, True,
                                      170.0 + i, 0.003, density))
                self.rows.append(_row("RRT*", "%s%d" % (density, i), 0, True,
                                      135.0 + i, 1.0, density))
        self.summary = summarise(self.rows)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_all_charts_render(self):
        from viz.charts import (chart_nodes, chart_path_length,
                                chart_success_profile, chart_time)
        for fn, args in [(chart_success_profile, (self.rows,)),
                         (chart_path_length, (self.summary,)),
                         (chart_time, (self.summary,)),
                         (chart_nodes, (self.summary,))]:
            out = os.path.join(self.tmp, "%s.png" % fn.__name__)
            self.assertTrue(os.path.exists(fn(*args, out)))

    def test_palette_is_the_validated_one(self):
        """Guard against a well-meaning edit swapping in unvalidated colours."""
        from viz.charts import PLANNER_COLORS
        self.assertEqual(PLANNER_COLORS["A*"], "#2a78d6")
        self.assertEqual(PLANNER_COLORS["RRT"], "#eb6834")
        self.assertEqual(PLANNER_COLORS["RRT*"], "#1baf7a")

    def test_path_plots_use_the_same_planner_colours(self):
        """Colour must identify the planner consistently across every figure."""
        from viz.charts import PLANNER_COLORS as chart_colors
        from viz.plot import PLANNER_COLORS as plot_colors
        self.assertEqual(chart_colors, plot_colors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
