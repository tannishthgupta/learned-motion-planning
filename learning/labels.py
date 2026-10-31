"""
Training labels: "which cells lie on a near-optimal path?"

For each problem we want the network to learn the region that a good sampler
should concentrate on. We define it exactly, on the feature grid:

    d_s(c)  = shortest-path distance from the start to cell c
    d_g(c)  = shortest-path distance from cell c to the goal
    d*      = d_s(goal) = the optimal start-to-goal distance

    label(c) = 1   if   d_s(c) + d_g(c) <= (1 + eps) * d*  + slack

d_s(c) + d_g(c) is the length of the best path that is *forced* through c, so
the label marks every cell that some path within eps of optimal passes
through. It is the grid version of the "informed set" from Informed RRT*
(Gammell et al., 2014), except that it follows obstacles instead of being a
plain ellipse. If several routes around an obstacle are nearly equally good,
all of them are labelled -- the label is not tied to one arbitrary path.

The two distance fields come from Dijkstra's algorithm (via SciPy) on a
16-connected grid. 16 neighbours (the 8 usual ones plus 8 "knight" moves) are
used instead of 8 because 8-connected distances are up to ~8% off from true
Euclidean distance depending on direction, which would bend the labels toward
the grid axes. 16-connected distances are within ~2.7%.

These labels are only a training target. The planner never trusts them: every
sample it draws is still collision-checked against the exact geometry.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# (dx, dy, cells that the straight move passes through on the way).
# A move is allowed only if the destination AND those intermediate cells are
# free, so a move can never cut an obstacle corner.
_MOVES = []
for _dx, _dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
    _MOVES.append((_dx, _dy, []))
for _dx in (1, -1):
    for _dy in (1, -1):
        _MOVES.append((_dx, _dy, [(_dx, 0), (0, _dy)]))
for _dx, _dy in [(2, 1), (2, -1), (-2, 1), (-2, -1)]:
    _MOVES.append((_dx, _dy, [(_dx // 2, 0), (_dx // 2, _dy)]))
for _dx, _dy in [(1, 2), (-1, 2), (1, -2), (-1, -2)]:
    _MOVES.append((_dx, _dy, [(0, _dy // 2), (_dx, _dy // 2)]))


def grid_graph(free):
    """Sparse adjacency matrix of the 16-connected graph over free cells.

    Edge weights are in cell units (multiply by the cell size for world units).
    """
    ny, nx = free.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    src_all, dst_all, w_all = [], [], []

    for dx, dy, via in _MOVES:
        # Window of source cells for which the destination is in the grid.
        x0, x1 = max(0, -dx), nx - max(0, dx)
        y0, y1 = max(0, -dy), ny - max(0, dy)
        if x0 >= x1 or y0 >= y1:
            continue
        ok = free[y0:y1, x0:x1] & free[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        for vx, vy in via:
            ok &= free[y0 + vy:y1 + vy, x0 + vx:x1 + vx]
        src = idx[y0:y1, x0:x1][ok]
        dst = idx[y0 + dy:y1 + dy, x0 + dx:x1 + dx][ok]
        src_all.append(src)
        dst_all.append(dst)
        w_all.append(np.full(src.shape, np.hypot(dx, dy)))

    src = np.concatenate(src_all)
    dst = np.concatenate(dst_all)
    w = np.concatenate(w_all)
    return csr_matrix((w, (src, dst)), shape=(ny * nx, ny * nx))


def near_optimal_region(free, start_cell, goal_cell, eps=0.02, slack=1.5):
    """Label every free cell that lies on a path within `eps` of optimal.

    Parameters
    ----------
    free : (ny, nx) bool
    start_cell, goal_cell : (col, row) integer cell indices
    eps : relative sub-optimality allowed (0.02 = paths up to 2% longer)
    slack : extra allowance in cells, absorbing discretisation error so that
            a thin optimal corridor is never labelled as empty

    Returns
    -------
    label : (ny, nx) bool, or None if the goal is unreachable on this grid
    d_opt : optimal distance in cell units (inf if unreachable)
    """
    ny, nx = free.shape
    sc = int(start_cell[1]) * nx + int(start_cell[0])
    gc = int(goal_cell[1]) * nx + int(goal_cell[0])
    flat_free = free.ravel()
    if not (flat_free[sc] and flat_free[gc]):
        return None, float("inf")

    graph = grid_graph(free)
    # Moves are symmetric, so one call gives distances from both endpoints.
    dist = dijkstra(graph, directed=True, indices=[sc, gc])
    d_s, d_g = dist[0], dist[1]
    d_opt = d_s[gc]
    if not np.isfinite(d_opt):
        return None, float("inf")

    through = d_s + d_g
    label = (through <= (1.0 + eps) * d_opt + slack) & flat_free
    return label.reshape(ny, nx), float(d_opt)
