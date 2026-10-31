"""
Turning a planning problem into a fixed-size image the network can read.

A problem is (world, start, goal). The network sees it as a stack of GRID x GRID
channels over the workspace:

    0  free      1 where the robot (a disc) can sit, 0 where it collides.
                 Uses the same robot-radius inflation as the Environment.
    1  start     a small Gaussian blob centred on the start.
    2  goal      a small Gaussian blob centred on the goal.
    3  d_start   straight-line distance from each cell to the start, scaled.
    4  d_goal    straight-line distance from each cell to the goal, scaled.

Channels 3-4 give every pixel global context ("how far am I from the
endpoints?") that a small convolutional network would otherwise need many
layers to compute. All five channels rotate and flip together with the map,
which is what makes the dihedral data augmentation in train.py valid.

Pure NumPy -- the same function is used to build training data and to build
the network input at planning time, so the two can never drift apart.
"""

import numpy as np

from env.grid import OccupancyGrid

#: Side length of the network's input/output grid.
GRID = 64

N_CHANNELS = 5

# Width of the start/goal blobs, in cells.
_BLOB_SIGMA = 1.5


def free_mask(env, size=GRID):
    """(size, size) bool array, True where the robot may sit at the cell centre."""
    if abs(env.width - env.height) > 1e-9:
        raise ValueError("learning features assume a square workspace")
    grid = OccupancyGrid(env, resolution=env.width / size)
    blocked = grid.blocked_mask()
    if blocked.shape != (size, size):
        raise RuntimeError("unexpected grid shape %r" % (blocked.shape,))
    return ~blocked


def to_cell_coords(env, p, size=GRID):
    """World point -> continuous (col, row) coordinates on the feature grid.

    Cell (i, j) has its centre at (i, j), so a point at the centre of the
    lower-left cell maps to (0, 0).
    """
    return (p[0] / env.width * size - 0.5, p[1] / env.height * size - 0.5)


def make_features(free, start_cell, goal_cell):
    """Build the (N_CHANNELS, size, size) float32 input for one problem.

    `free` is the bool mask; `start_cell` / `goal_cell` are continuous
    (col, row) coordinates from `to_cell_coords`.
    """
    size = free.shape[0]
    rows, cols = np.mgrid[0:size, 0:size].astype(np.float32)

    def dist(c):
        return np.hypot(cols - c[0], rows - c[1])

    ds = dist(start_cell)
    dg = dist(goal_cell)

    x = np.empty((N_CHANNELS, size, size), dtype=np.float32)
    x[0] = free
    x[1] = np.exp(-0.5 * (ds / _BLOB_SIGMA) ** 2)
    x[2] = np.exp(-0.5 * (dg / _BLOB_SIGMA) ** 2)
    # Divide by the grid diagonal so the values sit in roughly [0, 1].
    diag = size * np.sqrt(2.0)
    x[3] = ds / diag
    x[4] = dg / diag
    return x


def problem_features(env, start=None, goal=None, size=GRID):
    """Convenience wrapper: Environment (+ optional start/goal) -> (features, free)."""
    start = env.start if start is None else start
    goal = env.goal if goal is None else goal
    free = free_mask(env, size)
    x = make_features(free, to_cell_coords(env, start, size),
                      to_cell_coords(env, goal, size))
    return x, free
