"""
Uniform grid discretisation of a continuous Environment.

A* needs a discrete graph; RRT and RRT* do not. To keep the comparison fair,
this grid supplies *only* the node set (cell centres) and the neighbourhood
structure. Whether a cell may be occupied, and whether a move between two
cells is legal, is still decided by the Environment's own collision queries --
the exact same ones RRT and RRT* call.

Two consequences worth stating in the report:

  * A* never gets a cheaper or more permissive collision test than the
    sampling planners, so a success-rate gap is a real planner difference and
    not an artefact of the collision model.
  * Validity is resolved *lazily*, per cell, as the search touches it. A*
    therefore pays for exactly the states it expands, which is what makes its
    collision-check count comparable to the sampling planners'.

`blocked_mask()` is the one bulk operation, used for visualisation and for the
generator's reachability filter. It is vectorised with NumPy and deliberately
bypasses the Environment counters, since it is not part of any planner's work.
"""

import math

import numpy as np

from .obstacles import CircleObstacle, RectObstacle

# (dx, dy) offsets for 8-connectivity, paired with their step costs.
# Diagonal moves cost sqrt(2) so that grid distance approximates Euclidean
# distance -- without this the "path length" metric would be inflated by the
# staircase effect and A* would look artificially bad against RRT*.
_NEIGHBOURS_8 = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]

_NEIGHBOURS_4 = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]


class OccupancyGrid:
    def __init__(self, env, resolution=1.0, connectivity=8):
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")

        self.env = env
        self.resolution = float(resolution)
        self.connectivity = connectivity
        self.neighbour_offsets = _NEIGHBOURS_8 if connectivity == 8 else _NEIGHBOURS_4

        self.nx = max(1, int(math.floor(env.width / self.resolution)))
        self.ny = max(1, int(math.floor(env.height / self.resolution)))

        # Lazy validity cache: cell index -> bool. Populated on first touch.
        self._valid_cache = {}

    # ------------------------------------------------------------------
    # Index <-> world conversions
    # ------------------------------------------------------------------

    def cell_center(self, ix, iy):
        """World coordinates of the centre of cell (ix, iy)."""
        return ((ix + 0.5) * self.resolution, (iy + 0.5) * self.resolution)

    def world_to_cell(self, p):
        """Cell containing world point p, clamped to the grid extent."""
        ix = int(p[0] / self.resolution)
        iy = int(p[1] / self.resolution)
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        return (ix, iy)

    def in_grid(self, ix, iy):
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    # ------------------------------------------------------------------
    # Validity (lazy, and routed through the shared Environment queries)
    # ------------------------------------------------------------------

    def is_cell_valid(self, ix, iy):
        """True if the robot may sit at the centre of cell (ix, iy)."""
        if not self.in_grid(ix, iy):
            return False
        cached = self._valid_cache.get((ix, iy))
        if cached is None:
            cached = self.env.is_state_valid(self.cell_center(ix, iy))
            self._valid_cache[(ix, iy)] = cached
        return cached

    def is_move_valid(self, a_cell, b_cell):
        """True if the straight move between two cell centres is collision-free."""
        return self.env.is_motion_valid(self.cell_center(*a_cell),
                                        self.cell_center(*b_cell))

    def neighbours(self, ix, iy):
        """Yield ((nx, ny), step_cost) for in-grid, collision-free neighbours.

        Both the destination cell *and* the swept motion are checked, so a
        diagonal move can never cut the corner of an obstacle.
        """
        for dx, dy, cost in self.neighbour_offsets:
            jx, jy = ix + dx, iy + dy
            if not self.is_cell_valid(jx, jy):
                continue
            if not self.is_move_valid((ix, iy), (jx, jy)):
                continue
            yield (jx, jy), cost * self.resolution

    def nearest_valid_cell(self, p, max_radius_cells=10):
        """Cell nearest to world point p whose centre is collision-free.

        Start and goal are continuous coordinates that rarely land on a cell
        centre, so A* snaps them to the closest usable cell. The snap distance
        is added back into the reported path length by the planner.
        """
        ix0, iy0 = self.world_to_cell(p)
        if self.is_cell_valid(ix0, iy0):
            return (ix0, iy0)

        # Expanding ring search around the containing cell.
        for radius in range(1, max_radius_cells + 1):
            best, best_d = None, float("inf")
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only the shell of the square, not its interior.
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    ix, iy = ix0 + dx, iy0 + dy
                    if not self.is_cell_valid(ix, iy):
                        continue
                    c = self.cell_center(ix, iy)
                    d = math.hypot(c[0] - p[0], c[1] - p[1])
                    if d < best_d:
                        best, best_d = (ix, iy), d
            if best is not None:
                return best
        return None

    # ------------------------------------------------------------------
    # Bulk mask -- visualisation and reachability only, never planner work
    # ------------------------------------------------------------------

    def blocked_mask(self):
        """Boolean array of shape (ny, nx); True where the cell centre is blocked.

        Vectorised over all cells at once and does NOT increment the
        Environment's collision counters, because no planner performs this.
        """
        res = self.resolution
        xs = (np.arange(self.nx) + 0.5) * res
        ys = (np.arange(self.ny) + 0.5) * res
        gx, gy = np.meshgrid(xs, ys)           # both (ny, nx)

        r = self.env.robot_radius
        blocked = np.zeros((self.ny, self.nx), dtype=bool)

        # Workspace bounds: the robot disc must fit entirely inside.
        blocked |= (gx < r) | (gx > self.env.width - r)
        blocked |= (gy < r) | (gy > self.env.height - r)

        for o in self.env.obstacles:
            if isinstance(o, RectObstacle):
                xmin, ymin, xmax, ymax = o.box
                dx = np.maximum.reduce([xmin - gx, np.zeros_like(gx), gx - xmax])
                dy = np.maximum.reduce([ymin - gy, np.zeros_like(gy), gy - ymax])
                dist = np.hypot(dx, dy)
            elif isinstance(o, CircleObstacle):
                dist = np.maximum(np.hypot(gx - o.cx, gy - o.cy) - o.r, 0.0)
            else:
                raise TypeError("blocked_mask does not handle %r" % type(o))
            blocked |= (dist <= r)

        return blocked

    def __repr__(self):
        return "OccupancyGrid(%dx%d cells, res=%g, conn=%d)" % (
            self.nx, self.ny, self.resolution, self.connectivity)
