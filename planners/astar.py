"""
A* search on a uniform grid.

Reference: Hart, Nilsson & Raphael, "A Formal Basis for the Heuristic
Determination of Minimum Cost Paths", IEEE TSSC, 1968.

A* expands the open node minimising

    f(n) = g(n) + w * h(n)

where g is the cost accumulated from the start and h is an estimate of the
cost still to go. With w = 1 and an admissible, consistent h, the first time
the goal is popped its g-value is optimal -- for the discretised problem. That
last qualifier matters for the report: A* is optimal *on the grid it is given*,
not over the continuous workspace, so its path length is bounded below by the
true optimum and the gap shrinks as the resolution gets finer.

Heuristic: octile distance, which is the exact cost of an unobstructed
8-connected grid path. It is admissible (obstacles can only make things worse)
and consistent, and it dominates straight-line Euclidean distance, so it
expands strictly fewer nodes. Euclidean is kept as an option for comparison.
"""

import heapq
import math

from env.geometry import distance
from env.grid import OccupancyGrid

from .base import Planner

SQRT2 = math.sqrt(2.0)


def octile_distance(a, b, resolution):
    """Exact cost of an obstacle-free 8-connected path between two cells.

    Take as many diagonal steps as possible, then straight steps for the
    remainder: sqrt(2)*min(dx,dy) + (max(dx,dy) - min(dx,dy)).
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    lo, hi = (dx, dy) if dx < dy else (dy, dx)
    return resolution * (SQRT2 * lo + (hi - lo))


def euclidean_cell_distance(a, b, resolution):
    """Straight-line distance between two cells, in world units."""
    return resolution * math.hypot(a[0] - b[0], a[1] - b[1])


HEURISTICS = {
    "octile": octile_distance,
    "euclidean": euclidean_cell_distance,
}


class AStarPlanner(Planner):
    """Grid-based A*.

    Parameters
    ----------
    resolution : float
        Grid cell size in world units. This is *the* accuracy/cost knob:
        halving it quadruples the number of cells.
    connectivity : {4, 8}
        Move set. 8 gives much better path quality; 4 forces staircase paths.
    heuristic : {"octile", "euclidean"}
    weight : float
        Heuristic weight w. 1.0 is standard admissible A*. Values > 1 give
        weighted A*: faster, but the path may exceed the optimum by up to a
        factor of w. Kept for the discussion section, not used for headline
        results.
    """

    name = "A*"

    def __init__(self, resolution=1.0, connectivity=8, heuristic="octile",
                 weight=1.0, time_budget=10.0):
        super().__init__(time_budget=time_budget)
        if heuristic not in HEURISTICS:
            raise ValueError("heuristic must be one of %s" % sorted(HEURISTICS))
        self.resolution = float(resolution)
        self.connectivity = int(connectivity)
        self.heuristic_name = heuristic
        self.heuristic_fn = HEURISTICS[heuristic]
        self.weight = float(weight)

        # Exposed for visualisation after a run.
        self.grid = None
        self.expanded = set()      # cells popped from the open list
        self.came_from = {}

    # ------------------------------------------------------------------

    def plan(self, env, start=None, goal=None):
        start_pt, goal_pt = self._start_goal(env, start, goal)
        t0 = self._begin(env)

        self.grid = OccupancyGrid(env, self.resolution, self.connectivity)
        self.expanded = set()
        self.came_from = {}

        # Snap the continuous endpoints onto usable cells. The leftover
        # start->cell and cell->goal stubs are stitched back on at the end, so
        # the reported path really does run start to goal.
        start_cell = self.grid.nearest_valid_cell(start_pt)
        goal_cell = self.grid.nearest_valid_cell(goal_pt)
        if start_cell is None:
            return None, self._finish(env, t0, None, 0, 0, "no free cell near start")
        if goal_cell is None:
            return None, self._finish(env, t0, None, 0, 0, "no free cell near goal")

        res = self.grid.resolution
        w = self.weight
        h = lambda c: self.heuristic_fn(c, goal_cell, res)

        # Open list entries are (f, h, counter, cell). Ordering by h on an
        # f-tie prefers nodes closer to the goal, which noticeably reduces
        # expansions in open space; `counter` just makes ordering total so the
        # heap never has to compare two cell tuples.
        counter = 0
        g_score = {start_cell: 0.0}
        open_heap = [(w * h(start_cell), h(start_cell), counter, start_cell)]
        closed = set()

        iterations = 0
        found = False

        while open_heap:
            iterations += 1

            # Checking the clock every so often keeps the budget meaningful
            # without paying for a syscall on every single pop.
            if iterations % 512 == 0 and self._out_of_time(t0):
                return None, self._finish(env, t0, None, iterations,
                                          len(self.expanded), "time budget exceeded")

            _, _, _, current = heapq.heappop(open_heap)

            # Lazy deletion: a cell can sit in the heap more than once if it
            # was re-reached with a lower g. Skip the stale copies.
            if current in closed:
                continue
            closed.add(current)
            self.expanded.add(current)

            if current == goal_cell:
                found = True
                break

            for neighbour, step_cost in self.grid.neighbours(*current):
                if neighbour in closed:
                    continue
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbour, float("inf")):
                    g_score[neighbour] = tentative_g
                    self.came_from[neighbour] = current
                    counter += 1
                    hn = h(neighbour)
                    heapq.heappush(open_heap,
                                   (tentative_g + w * hn, hn, counter, neighbour))

        if not found:
            return None, self._finish(env, t0, None, iterations, len(self.expanded),
                                      "open list exhausted (goal unreachable on grid)")

        path = self._reconstruct(env, start_cell, goal_cell, start_pt, goal_pt)
        if path is None:
            return None, self._finish(env, t0, None, iterations, len(self.expanded),
                                      "could not connect endpoints to grid path")

        return path, self._finish(env, t0, path, iterations, len(self.expanded))

    # ------------------------------------------------------------------

    def _reconstruct(self, env, start_cell, goal_cell, start_pt, goal_pt):
        """Walk parent pointers back, then stitch the true endpoints on.

        Returns None if either stub is itself in collision, which can happen
        when the true endpoint sits in a pocket its own cell centre cannot
        reach in a straight line.
        """
        cells = [goal_cell]
        while cells[-1] != start_cell:
            cells.append(self.came_from[cells[-1]])
        cells.reverse()

        path = [self.grid.cell_center(*c) for c in cells]

        # Front stub: true start -> first cell centre.
        if distance(start_pt, path[0]) > 1e-9:
            if not env.is_motion_valid(start_pt, path[0]):
                return None
            path.insert(0, start_pt)

        # Back stub: last cell centre -> true goal.
        if distance(goal_pt, path[-1]) > 1e-9:
            if not env.is_motion_valid(path[-1], goal_pt):
                return None
            path.append(goal_pt)

        return path

    def __repr__(self):
        return ("AStarPlanner(resolution=%g, connectivity=%d, heuristic=%r, "
                "weight=%g)" % (self.resolution, self.connectivity,
                                self.heuristic_name, self.weight))
