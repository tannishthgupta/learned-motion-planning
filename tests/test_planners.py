"""
Correctness tests for the world model and the three planners.

Run with:  python -m unittest discover -s tests -v

These are not smoke tests. Each one checks a property the project's claims
actually depend on -- if the report says "RRT* converges toward the optimum",
there is a test here that measures it. The recurring theme is that a planner
must never report success on a path that is not genuinely collision-free,
since that would silently corrupt every metric downstream.
"""

import math
import unittest

from env import (CircleObstacle, Environment, RectObstacle, get_scenario,
                 is_reachable, solvable_random_environment)
from env import geometry as geo
from env.grid import OccupancyGrid
from planners import AStarPlanner, RRTPlanner, RRTStarPlanner, Tree, verify_path
from planners.astar import octile_distance
from planners.rrt import steer


class TestGeometry(unittest.TestCase):
    """The analytic primitives every collision check is built on."""

    def test_point_segment_distance(self):
        # Perpendicular foot falls inside the segment.
        self.assertAlmostEqual(geo.point_segment_distance((0, 3), (-5, 0), (5, 0)), 3.0)
        # Foot falls beyond an endpoint -> distance to that endpoint.
        self.assertAlmostEqual(geo.point_segment_distance((10, 0), (-5, 0), (5, 0)), 5.0)
        # Degenerate segment behaves as a point.
        self.assertAlmostEqual(geo.point_segment_distance((3, 4), (0, 0), (0, 0)), 5.0)

    def test_segments_intersect(self):
        self.assertTrue(geo.segments_intersect((0, 0), (2, 2), (0, 2), (2, 0)))
        self.assertFalse(geo.segments_intersect((0, 0), (1, 1), (2, 2), (3, 3)))
        # Touching at an endpoint counts as intersecting.
        self.assertTrue(geo.segments_intersect((0, 0), (1, 1), (1, 1), (2, 0)))

    def test_segment_aabb_distance(self):
        box = (1, 1, 2, 2)
        # Passing straight through the box.
        self.assertEqual(geo.segment_aabb_distance((0, 0), (3, 3), box), 0.0)
        # Running parallel one unit above it.
        self.assertAlmostEqual(geo.segment_aabb_distance((0, 3), (3, 3), box), 1.0)
        # An endpoint inside the box.
        self.assertEqual(geo.segment_aabb_distance((1.5, 1.5), (9, 9), box), 0.0)


class TestCollisionModel(unittest.TestCase):
    """The Environment's feasibility queries."""

    def test_robot_radius_inflates_obstacles(self):
        env = Environment(10, 10, [RectObstacle(4, 0, 2, 10)], (1, 5), (9, 5),
                          robot_radius=1.0)
        # A point 0.5 from the wall is invalid for a radius-1.0 robot, even
        # though the point itself is outside the obstacle.
        self.assertFalse(env.is_state_valid((3.5, 5)))
        self.assertTrue(env.is_state_valid((2.5, 5)))

    def test_no_tunnelling_through_thin_obstacle(self):
        """A long edge must not skip over a thin wall.

        This is the failure mode of sampled (as opposed to exact) motion
        checking, and it would let the sampling planners walk through walls.
        """
        env = Environment(100, 100, [RectObstacle(49.9, 0, 0.2, 100)],
                          (5, 50), (95, 50), robot_radius=0.0)
        self.assertFalse(env.is_motion_valid((5, 50), (95, 50)))

    def test_bounds_respect_robot_radius(self):
        env = Environment(10, 10, [], (5, 5), (6, 6), robot_radius=1.0)
        self.assertFalse(env.is_state_valid((0.5, 5)))
        self.assertTrue(env.is_state_valid((1.5, 5)))

    def test_broad_phase_matches_exact_test(self):
        """The bounding-box early-out must never change an answer.

        Compares the optimised path against a brute-force exact evaluation on
        many random segments -- the guarantee that the speed-up is not an
        approximation.
        """
        import random
        rng = random.Random(7)
        env = get_scenario("cluttered_discs")
        r = env.robot_radius
        for _ in range(400):
            a = (rng.uniform(0, 100), rng.uniform(0, 100))
            b = (rng.uniform(0, 100), rng.uniform(0, 100))
            fast = env.is_motion_valid(a, b)
            exact = (env.in_bounds(a) and env.in_bounds(b) and
                     all(o.clearance_to_segment(a, b) > r for o in env.obstacles))
            self.assertEqual(fast, exact, "mismatch on segment %s -> %s" % (a, b))


class TestTree(unittest.TestCase):
    """The shared RRT/RRT* tree, especially rewiring."""

    def test_nearest_and_near(self):
        t = Tree((0, 0))
        t.add_node((10, 0), 0, 10.0)
        t.add_node((1, 1), 0, math.sqrt(2))
        self.assertEqual(t.nearest((1.2, 1.2)), 2)
        self.assertEqual(sorted(t.near((0, 0), 2.0)), [0, 2])

    def test_rewire_propagates_cost_to_descendants(self):
        """Re-parenting must shift the whole subtree's cost-to-come.

        Without propagation the descendants keep stale costs, and RRT* then
        rejects genuinely better rewirings later on.
        """
        t = Tree((0, 0))
        a = t.add_node((10, 0), 0, 10.0)
        b = t.add_node((20, 0), a, 20.0)
        c = t.add_node((30, 0), b, 30.0)

        # Re-attach `a` via a cheaper route: cost 10.0 -> 4.0, delta -6.
        t.set_parent(a, 0, 4.0)
        self.assertAlmostEqual(t.cost[a], 4.0)
        self.assertAlmostEqual(t.cost[b], 14.0)
        self.assertAlmostEqual(t.cost[c], 24.0)

    def test_path_to_root(self):
        t = Tree((0, 0))
        a = t.add_node((1, 0), 0, 1.0)
        b = t.add_node((2, 0), a, 2.0)
        self.assertEqual(t.path_to_root(b), [(0, 0), (1, 0), (2, 0)])


class TestSteer(unittest.TestCase):
    def test_respects_step_size(self):
        p = steer((0, 0), (100, 0), 5.0)
        self.assertAlmostEqual(p[0], 5.0)

    def test_returns_target_when_within_reach(self):
        # Must land exactly on the target, or the tree can never reach the goal.
        self.assertEqual(steer((0, 0), (3, 4), 10.0), (3, 4))


class TestAStar(unittest.TestCase):

    def test_finds_optimal_path_on_empty_world(self):
        """On an empty world the optimum is the straight line.

        A* is optimal *on its grid*, so allow the discretisation error but
        require it to be small.
        """
        env = get_scenario("empty")
        path, m = AStarPlanner(resolution=1.0).plan(env)
        optimal = geo.distance(env.start, env.goal)
        self.assertTrue(m["success"])
        self.assertLess(m["path_length"], optimal * 1.05)
        self.assertGreaterEqual(m["path_length"], optimal - 1e-9)

    def test_matches_dijkstra(self):
        """A*'s cost must equal Dijkstra's on the same grid.

        Dijkstra is A* with h = 0, so any disagreement means the heuristic is
        inadmissible. This is the real test of `octile_distance`.
        """
        env = get_scenario("cluttered_discs")

        _, m_astar = AStarPlanner(resolution=2.0, heuristic="octile").plan(env)
        # weight=0 zeroes the heuristic, turning A* into Dijkstra.
        _, m_dijkstra = AStarPlanner(resolution=2.0, weight=0.0).plan(env)

        self.assertTrue(m_astar["success"] and m_dijkstra["success"])
        self.assertAlmostEqual(m_astar["path_length"], m_dijkstra["path_length"], places=6)
        # And the heuristic must actually be doing work.
        self.assertLess(m_astar["nodes"], m_dijkstra["nodes"])

    def test_octile_is_admissible_vs_euclidean(self):
        """Octile never underestimates less than Euclidean, so it expands fewer."""
        env = get_scenario("cluttered_discs")
        _, m_oct = AStarPlanner(resolution=2.0, heuristic="octile").plan(env)
        _, m_euc = AStarPlanner(resolution=2.0, heuristic="euclidean").plan(env)
        self.assertAlmostEqual(m_oct["path_length"], m_euc["path_length"], places=6)
        self.assertLessEqual(m_oct["nodes"], m_euc["nodes"])

    def test_octile_distance_values(self):
        self.assertAlmostEqual(octile_distance((0, 0), (3, 0), 1.0), 3.0)
        self.assertAlmostEqual(octile_distance((0, 0), (3, 3), 1.0), 3 * math.sqrt(2))
        self.assertAlmostEqual(octile_distance((0, 0), (5, 3), 1.0), 3 * math.sqrt(2) + 2)

    def test_no_corner_cutting(self):
        """A diagonal move must not slip between two touching obstacles."""
        obstacles = [RectObstacle(5, 0, 5, 5), RectObstacle(0, 5, 5, 5)]
        env = Environment(10, 10, obstacles, (2.5, 2.5), (7.5, 7.5), robot_radius=0.0)
        path, m = AStarPlanner(resolution=1.0).plan(env)
        # The only route is diagonally through the shared corner, which is
        # blocked -- so the planner must report failure rather than cut it.
        if m["success"]:
            ok, why = verify_path(env, path)
            self.assertTrue(ok, why)


class TestRRT(unittest.TestCase):

    def test_finds_path_and_it_is_valid(self):
        env = get_scenario("cluttered_discs")
        path, m = RRTPlanner(seed=1, max_iters=20000).plan(env)
        self.assertTrue(m["success"], m["reason"])
        ok, why = verify_path(env, path)
        self.assertTrue(ok, why)

    def test_is_deterministic_for_a_given_seed(self):
        """Reproducibility: the benchmark's per-trial seeds must mean something."""
        env = get_scenario("cluttered_discs")
        p1, m1 = RRTPlanner(seed=99).plan(env)
        p2, m2 = RRTPlanner(seed=99).plan(env)
        self.assertEqual(p1, p2)
        self.assertEqual(m1["iterations"], m2["iterations"])

    def test_different_seeds_give_different_paths(self):
        env = get_scenario("cluttered_discs")
        p1, _ = RRTPlanner(seed=1).plan(env)
        p2, _ = RRTPlanner(seed=2).plan(env)
        self.assertNotEqual(p1, p2)

    def test_reports_failure_rather_than_a_bad_path(self):
        """A goal walled off entirely must yield failure, not a path through the wall."""
        env = Environment(100, 100, [RectObstacle(40, 0, 20, 100)],
                          (10, 50), (90, 50), robot_radius=1.0)
        path, m = RRTPlanner(seed=5, max_iters=2000, time_budget=5.0).plan(env)
        self.assertFalse(m["success"])
        self.assertIsNone(path)


class TestRRTStar(unittest.TestCase):

    def test_finds_valid_path(self):
        env = get_scenario("cluttered_discs")
        path, m = RRTStarPlanner(seed=1, max_iters=3000).plan(env)
        self.assertTrue(m["success"], m["reason"])
        ok, why = verify_path(env, path)
        self.assertTrue(ok, why)

    def test_converges_toward_the_optimum_on_empty_world(self):
        """The headline claim: RRT* approaches the straight-line optimum.

        On an empty world the optimum is known exactly, so this measures the
        optimality gap rather than merely asserting improvement.
        """
        env = get_scenario("empty")
        optimal = geo.distance(env.start, env.goal)
        _, m = RRTStarPlanner(seed=3, max_iters=6000, time_budget=60.0).plan(env)
        self.assertTrue(m["success"])
        self.assertLess(m["path_length"], optimal * 1.05,
                        "RRT* path %.2f is more than 5%% above the optimum %.2f"
                        % (m["path_length"], optimal))

    def test_refines_its_own_first_solution(self):
        """Rewiring must improve on the first path found -- that is the whole point."""
        env = get_scenario("maze")
        _, m = RRTStarPlanner(seed=4, max_iters=6000, time_budget=60.0).plan(env)
        self.assertTrue(m["success"])
        self.assertLess(m["path_length"], m["first_solution_length"])

    def test_beats_rrt_on_path_length(self):
        """Averaged over seeds, RRT* must produce shorter paths than RRT.

        Averaged rather than per-seed: RRT is random, and on a lucky seed it
        can match RRT* on a single run. The claim is statistical.
        """
        env = get_scenario("cluttered_discs")
        rrt_lengths, star_lengths = [], []
        for seed in range(5):
            _, m1 = RRTPlanner(seed=seed, max_iters=20000).plan(env)
            _, m2 = RRTStarPlanner(seed=seed, max_iters=3000, time_budget=60.0).plan(env)
            if m1["success"]:
                rrt_lengths.append(m1["path_length"])
            if m2["success"]:
                star_lengths.append(m2["path_length"])

        self.assertTrue(rrt_lengths and star_lengths)
        mean_rrt = sum(rrt_lengths) / len(rrt_lengths)
        mean_star = sum(star_lengths) / len(star_lengths)
        self.assertLess(mean_star, mean_rrt,
                        "RRT* mean %.2f should beat RRT mean %.2f" % (mean_star, mean_rrt))

    def test_cost_history_is_monotonically_decreasing(self):
        """The best-cost curve may never go up: rewiring only ever improves."""
        env = get_scenario("maze")
        planner = RRTStarPlanner(seed=6, max_iters=4000, time_budget=60.0)
        planner.plan(env)
        costs = [c for _, _, c in planner.cost_history]
        self.assertGreater(len(costs), 1, "expected the path cost to improve at least once")
        for earlier, later in zip(costs, costs[1:]):
            self.assertLessEqual(later, earlier + 1e-9)


class TestAllPlannersAgree(unittest.TestCase):
    """Cross-planner invariants that hold on every world."""

    def test_every_returned_path_is_collision_free(self):
        for scenario in ["empty", "cluttered_discs", "maze", "narrow_gap", "bug_trap"]:
            for planner in [AStarPlanner(time_budget=30.0),
                            RRTPlanner(seed=11, max_iters=20000, time_budget=30.0),
                            RRTStarPlanner(seed=11, max_iters=3000, time_budget=30.0)]:
                env = get_scenario(scenario)
                path, m = planner.plan(env)
                if m["success"]:
                    ok, why = verify_path(env, path)
                    self.assertTrue(ok, "%s on %s: %s" % (m["planner"], scenario, why))

    def test_paths_start_at_start_and_end_at_goal(self):
        env = get_scenario("cluttered_discs")
        for planner in [AStarPlanner(), RRTPlanner(seed=2, max_iters=20000),
                        RRTStarPlanner(seed=2, max_iters=2000)]:
            path, m = planner.plan(env)
            if m["success"]:
                self.assertAlmostEqual(geo.distance(path[0], env.start), 0.0, places=9)
                self.assertAlmostEqual(geo.distance(path[-1], env.goal), 0.0, places=9)

    def test_metrics_have_the_full_schema(self):
        from planners import METRIC_FIELDS
        env = get_scenario("empty")
        for planner in [AStarPlanner(), RRTPlanner(seed=1), RRTStarPlanner(seed=1, max_iters=500)]:
            _, m = planner.plan(env)
            self.assertEqual(set(m), set(METRIC_FIELDS))


class TestEnvironmentGeneration(unittest.TestCase):

    def test_generated_worlds_are_solvable(self):
        for density in (0.1, 0.2, 0.3):
            env, _ = solvable_random_environment(density=density, seed=5)
            self.assertTrue(is_reachable(env))
            self.assertTrue(env.is_state_valid(env.start))
            self.assertTrue(env.is_state_valid(env.goal))

    def test_serialisation_round_trip(self):
        env = get_scenario("cluttered_discs")
        clone = Environment.from_dict(env.to_dict())
        self.assertEqual(len(clone.obstacles), len(env.obstacles))
        self.assertAlmostEqual(clone.obstacle_density(), env.obstacle_density())
        # And the reconstructed world must answer collision queries identically.
        self.assertEqual(clone.is_motion_valid(env.start, env.goal),
                         env.is_motion_valid(env.start, env.goal))

    def test_validate_rejects_a_blocked_start(self):
        env = Environment(10, 10, [CircleObstacle(5, 5, 3)], (5, 5), (9, 9))
        with self.assertRaises(ValueError):
            env.validate()

    def test_grid_blocked_mask_agrees_with_state_validity(self):
        """The vectorised mask must match the exact per-point query."""
        env = get_scenario("cluttered_discs")
        grid = OccupancyGrid(env, resolution=2.0)
        mask = grid.blocked_mask()
        for iy in range(grid.ny):
            for ix in range(grid.nx):
                expected = not env.is_state_valid(grid.cell_center(ix, iy))
                self.assertEqual(bool(mask[iy, ix]), expected,
                                 "cell (%d,%d) disagrees" % (ix, iy))


if __name__ == "__main__":
    unittest.main(verbosity=2)
