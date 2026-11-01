"""
Robot-arm demo: plan for planar arms with the unchanged RRT / RRT* planners.

    python arm_demo.py

Writes to results/arm/:
    two_link_cspace.png    2-joint arm: the world and its joint space side by side
    random_<dof>dof.png    a random world, solved by RRT* for a 4-joint arm

Joint-space planning settings are shared by every arm run in the project:
a steering step of 0.5 rad and a goal tolerance of 0.5 rad.
"""

import os

from arm import random_arm_problem, scenario_two_link
from arm.plot import figure_arm_path, figure_two_link
from planners import RRTStarPlanner, verify_path

OUT = os.path.join("results", "arm")
ARM_SETTINGS = dict(step_size=0.5, goal_tolerance=0.5, goal_bias=0.05, time_budget=120.0)


def run(env, max_iters, seed=0):
    planner = RRTStarPlanner(max_iters=max_iters, seed=seed, **ARM_SETTINGS)
    path, m = planner.plan(env)
    if not m["success"]:
        raise SystemExit("%s: no path in %d iterations" % (env.name, max_iters))
    ok, why = verify_path(env, path)
    assert ok, why
    print("%-12s %d joints: first path %.2f rad (iteration %d), final %.2f rad, %.1f s"
          % (env.name, env.arm.dof, m["first_solution_length"],
             m["first_solution_iteration"], m["path_length"], m["time_sec"]))
    return planner, path, m


def main():
    os.makedirs(OUT, exist_ok=True)

    env = scenario_two_link()
    planner, path, _ = run(env, max_iters=1500)
    print("  ->", figure_two_link(env, planner, path,
                                  "A 2-joint arm, planned in joint space with RRT*",
                                  os.path.join(OUT, "two_link_cspace.png")))

    env = random_arm_problem(4, seed=1)
    planner, path, m = run(env, max_iters=3000)
    print("  ->", figure_arm_path(
        env, path, "A 4-joint arm in a random world (RRT*)",
        "Green = start, red = goal, teal = poses along the path.\n"
        "Path cost %.2f rad of total joint rotation." % m["path_length"],
        os.path.join(OUT, "random_4dof.png")))


if __name__ == "__main__":
    main()
