"""
Environment generation.

Two families of worlds:

  * Random worlds at a controlled obstacle density (sparse -> cluttered).
    Used for the headline comparison, because density is the independent
    variable we sweep.

  * Fixed, hand-designed scenarios (narrow gap, bug trap, ...). These are not
    random: each one isolates a known behavioural difference between the
    planners, so the report can discuss concrete failure modes rather than
    purely statistical ones.

Every generated world is validated and -- optionally -- filtered for
reachability, so that a planner failure is a planner failure and not an
unsolvable instance.
"""

import random

import numpy as np

from .environment import Environment
from .grid import OccupancyGrid
from .obstacles import CircleObstacle, RectObstacle

# Default workspace used throughout the project.
DEFAULT_WIDTH = 100.0
DEFAULT_HEIGHT = 100.0
DEFAULT_ROBOT_RADIUS = 1.0
DEFAULT_START = (5.0, 5.0)
DEFAULT_GOAL = (95.0, 95.0)

# Density levels swept in the benchmark, labelled for plots and tables.
DENSITY_LEVELS = {
    "sparse": 0.10,
    "moderate": 0.20,
    "cluttered": 0.30,
}


# ----------------------------------------------------------------------------
# Reachability filter
# ----------------------------------------------------------------------------

def is_reachable(env, resolution=0.5):
    """Flood-fill test: is the goal reachable from the start at all?

    Runs on a grid finer than any planner uses, so it is a permissive filter:
    it rejects worlds that are definitely unsolvable without pre-judging what a
    given planner can find. A planner that fails on a world passing this test
    has genuinely failed.
    """
    grid = OccupancyGrid(env, resolution=resolution, connectivity=8)
    blocked = grid.blocked_mask()

    start = grid.world_to_cell(env.start)
    goal = grid.world_to_cell(env.goal)
    if blocked[start[1], start[0]] or blocked[goal[1], goal[0]]:
        return False

    # Iterative flood fill (deep recursion would overflow on a fine grid).
    seen = np.zeros_like(blocked)
    seen[start[1], start[0]] = True
    stack = [start]
    while stack:
        ix, iy = stack.pop()
        if (ix, iy) == goal:
            return True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                jx, jy = ix + dx, iy + dy
                if not (0 <= jx < grid.nx and 0 <= jy < grid.ny):
                    continue
                if blocked[jy, jx] or seen[jy, jx]:
                    continue
                seen[jy, jx] = True
                stack.append((jx, jy))
    return False


# ----------------------------------------------------------------------------
# Random worlds
# ----------------------------------------------------------------------------

def random_environment(density=0.2, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                       start=DEFAULT_START, goal=DEFAULT_GOAL,
                       robot_radius=DEFAULT_ROBOT_RADIUS, seed=None,
                       shape_mix=0.6, min_size=5.0, max_size=16.0,
                       endpoint_clearance=4.0, max_attempts=2000, name=None):
    """Scatter obstacles until the target area coverage is reached.

    `density` is the fraction of workspace area covered by obstacles.
    `shape_mix` is the probability that a given obstacle is a rectangle.

    Obstacles that would swallow the start or goal (within
    `endpoint_clearance`) are rejected and re-drawn, so the instance stays
    well-posed. Overlapping obstacles are allowed -- they are what produce the
    concave traps that distinguish the planners.
    """
    rng = random.Random(seed)
    target_area = density * width * height

    obstacles = []
    area = 0.0
    attempts = 0

    while area < target_area and attempts < max_attempts:
        attempts += 1

        if rng.random() < shape_mix:
            w = rng.uniform(min_size, max_size)
            h = rng.uniform(min_size, max_size)
            x = rng.uniform(0, width - w)
            y = rng.uniform(0, height - h)
            obs = RectObstacle(x, y, w, h)
            obs_area = w * h
        else:
            r = rng.uniform(min_size, max_size) / 2.0
            cx = rng.uniform(r, width - r)
            cy = rng.uniform(r, height - r)
            obs = CircleObstacle(cx, cy, r)
            obs_area = np.pi * r ** 2

        # Keep the endpoints usable.
        margin = robot_radius + endpoint_clearance
        if obs.clearance_to_point(start) <= margin:
            continue
        if obs.clearance_to_point(goal) <= margin:
            continue

        obstacles.append(obs)
        area += obs_area

    if name is None:
        name = "random_d%.2f_s%s" % (density, seed)

    return Environment(width, height, obstacles, start, goal,
                       robot_radius=robot_radius, name=name).validate()


def solvable_random_environment(density=0.2, seed=None, max_resamples=50, **kwargs):
    """Draw random worlds until one is reachable; returns (env, n_rejected).

    Raises if no solvable world turns up, which would mean the requested
    density is simply too high for this workspace.
    """
    rejected = 0
    for k in range(max_resamples):
        # Derive a fresh but reproducible seed for each retry.
        trial_seed = None if seed is None else seed * 1000 + k
        env = random_environment(density=density, seed=trial_seed, **kwargs)
        if is_reachable(env):
            return env, rejected
        rejected += 1
    raise RuntimeError("no solvable world at density %.2f after %d attempts"
                       % (density, max_resamples))


def generate_suite(densities=None, n_per_density=10, seed=0, **kwargs):
    """Build the benchmark world set: n solvable worlds per density level.

    All three planners run on this identical list, which is the whole point --
    a fair comparison requires the same instances, not merely the same
    distribution.
    """
    if densities is None:
        densities = DENSITY_LEVELS

    suite = []
    for label, density in densities.items():
        for i in range(n_per_density):
            env, _ = solvable_random_environment(
                density=density, seed=seed + 7919 * i + 104729 * len(label),
                **kwargs)
            env.name = "%s_%02d" % (label, i)
            env.density_label = label
            env.density_target = density
            suite.append(env)
    return suite


# ----------------------------------------------------------------------------
# Fixed scenarios -- each isolates one known planner behaviour
# ----------------------------------------------------------------------------

def scenario_empty():
    """No obstacles. Sanity baseline: the optimum is the straight line."""
    return Environment(DEFAULT_WIDTH, DEFAULT_HEIGHT, [], DEFAULT_START,
                       DEFAULT_GOAL, DEFAULT_ROBOT_RADIUS, name="empty").validate()


def scenario_narrow_gap(gap=6.0):
    """A wall across the world with one small doorway.

    The classic sampling-based failure mode: the chance of sampling into the
    gap is tiny, so RRT needs many iterations, while A* walks straight to it.
    Tests the narrow-passage weakness.
    """
    w, h = DEFAULT_WIDTH, DEFAULT_HEIGHT
    mid = h / 2.0
    lower = mid - gap / 2.0
    upper = mid + gap / 2.0
    obstacles = [
        RectObstacle(45.0, 0.0, 10.0, lower),
        RectObstacle(45.0, upper, 10.0, h - upper),
    ]
    return Environment(w, h, obstacles, DEFAULT_START, DEFAULT_GOAL,
                       DEFAULT_ROBOT_RADIUS, name="narrow_gap").validate()


def scenario_bug_trap():
    """A C-shaped pocket opening away from the goal, with the start inside.

    Greedy goal-biased growth pushes the tree into the closed end, so the
    planner must explore backwards to escape. Penalises naive goal bias.
    """
    obstacles = [
        RectObstacle(10.0, 10.0, 40.0, 5.0),    # bottom of the C
        RectObstacle(10.0, 35.0, 40.0, 5.0),    # top of the C
        RectObstacle(45.0, 10.0, 5.0, 30.0),    # closed right end
        RectObstacle(60.0, 0.0, 8.0, 70.0),     # outer wall to route around
    ]
    return Environment(DEFAULT_WIDTH, DEFAULT_HEIGHT, obstacles,
                       start=(20.0, 25.0), goal=(90.0, 30.0),
                       robot_radius=DEFAULT_ROBOT_RADIUS,
                       name="bug_trap").validate()


def scenario_maze():
    """Staggered walls forming a serpentine corridor.

    Long detours with no straight-line shortcut: separates path *quality*
    (RRT* converging toward the optimum) from mere feasibility (plain RRT).
    """
    obstacles = [
        RectObstacle(20.0, 0.0, 6.0, 70.0),
        RectObstacle(40.0, 30.0, 6.0, 70.0),
        RectObstacle(60.0, 0.0, 6.0, 70.0),
        RectObstacle(80.0, 30.0, 6.0, 70.0),
    ]
    return Environment(DEFAULT_WIDTH, DEFAULT_HEIGHT, obstacles,
                       start=(5.0, 85.0), goal=(95.0, 10.0),
                       robot_radius=DEFAULT_ROBOT_RADIUS,
                       name="maze").validate()


def scenario_cluttered_discs(n=25, seed=3):
    """Dense field of round obstacles -- many homotopy classes, no traps.

    The setting where RRT* has the most room to improve on RRT, since good
    detours exist in every direction and rewiring can find them.
    """
    rng = random.Random(seed)
    obstacles = []
    while len(obstacles) < n:
        r = rng.uniform(3.0, 7.0)
        cx = rng.uniform(r, DEFAULT_WIDTH - r)
        cy = rng.uniform(r, DEFAULT_HEIGHT - r)
        c = CircleObstacle(cx, cy, r)
        if c.clearance_to_point(DEFAULT_START) <= 5.0:
            continue
        if c.clearance_to_point(DEFAULT_GOAL) <= 5.0:
            continue
        obstacles.append(c)
    return Environment(DEFAULT_WIDTH, DEFAULT_HEIGHT, obstacles, DEFAULT_START,
                       DEFAULT_GOAL, DEFAULT_ROBOT_RADIUS,
                       name="cluttered_discs").validate()


SCENARIOS = {
    "empty": scenario_empty,
    "narrow_gap": scenario_narrow_gap,
    "bug_trap": scenario_bug_trap,
    "maze": scenario_maze,
    "cluttered_discs": scenario_cluttered_discs,
}


def get_scenario(name):
    """Build a fixed scenario by name."""
    if name not in SCENARIOS:
        raise KeyError("unknown scenario %r; available: %s"
                       % (name, ", ".join(sorted(SCENARIOS))))
    return SCENARIOS[name]()
