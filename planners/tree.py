"""
The search tree shared by RRT and RRT*.

Nodes are stored in flat parallel arrays rather than linked objects:

    coords[i]    -- (x, y) of node i
    parent[i]    -- index of the parent of node i (-1 for the root)
    cost[i]      -- cost-to-come from the root, along tree edges
    children[i]  -- indices whose parent is i

The flat layout exists for one reason: `nearest` and `near` are the inner loop
of both planners and are called once per iteration over a tree that grows to
thousands of nodes. Holding the coordinates in one contiguous NumPy array lets
both queries run as a single vectorised distance computation instead of a
Python loop.

This is a brute-force O(n) nearest-neighbour search, not a k-d tree. That is a
deliberate choice: it keeps the code readable and traceable for the viva, and
a k-d tree would need rebuilding or rebalancing on every insertion anyway. The
cost is documented in the report as a known scaling limitation.
"""

import numpy as np

_INITIAL_CAPACITY = 1024


class Tree:
    def __init__(self, root):
        # Dimension comes from the root: 2 for (x, y), N for an N-joint arm.
        self.dim = len(root)
        self._coords = np.empty((_INITIAL_CAPACITY, self.dim), dtype=float)
        self.parent = []
        self.cost = []
        self.children = []
        self.n = 0
        self.add_node(root, parent=-1, cost=0.0)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_node(self, p, parent, cost):
        """Append a node and return its index."""
        if self.n == self._coords.shape[0]:
            # Amortised doubling: growing by one would make building an
            # n-node tree O(n^2) in copying alone.
            self._coords = np.resize(self._coords, (self._coords.shape[0] * 2, self.dim))

        self._coords[self.n] = p
        self.parent.append(parent)
        self.cost.append(float(cost))
        self.children.append([])
        if parent >= 0:
            self.children[parent].append(self.n)

        self.n += 1
        return self.n - 1

    @property
    def coords(self):
        """View of the live coordinates, shape (n, dim)."""
        return self._coords[:self.n]

    def point(self, i):
        if self.dim == 2:
            return (self._coords[i, 0], self._coords[i, 1])
        return tuple(self._coords[i])

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def nearest(self, p):
        """Index of the node closest to p (Euclidean)."""
        d2 = self._squared_distances(p)
        return int(np.argmin(d2))

    def near(self, p, radius):
        """Indices of all nodes within `radius` of p."""
        d2 = self._squared_distances(p)
        return np.flatnonzero(d2 <= radius * radius).tolist()

    def _squared_distances(self, p):
        """Squared distance from p to every node. Squared to avoid n sqrts."""
        delta = self.coords - np.asarray(p, dtype=float)
        return np.einsum("ij,ij->i", delta, delta)

    # ------------------------------------------------------------------
    # Rewiring (RRT* only)
    # ------------------------------------------------------------------

    def set_parent(self, idx, new_parent, new_cost):
        """Re-attach node `idx` under `new_parent` and propagate the cost change.

        Propagation matters: after rewiring, every descendant of `idx` reaches
        the root through the new, cheaper route, so their cost-to-come values
        must shift by the same delta. Skipping this is a common shortcut in
        simple RRT* implementations, and it leaves stale costs that cause the
        algorithm to reject genuinely better rewirings later on.
        """
        old_parent = self.parent[idx]
        if old_parent >= 0:
            self.children[old_parent].remove(idx)

        self.parent[idx] = new_parent
        self.children[new_parent].append(idx)

        delta = new_cost - self.cost[idx]
        self.cost[idx] = new_cost

        if delta == 0.0:
            return

        # Iterative walk over the subtree; recursion would risk overflowing
        # the stack on the long chains RRT* tends to build.
        stack = list(self.children[idx])
        while stack:
            child = stack.pop()
            self.cost[child] += delta
            stack.extend(self.children[child])

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def path_to_root(self, idx):
        """Waypoints from the root down to node `idx`, root first."""
        out = []
        while idx != -1:
            out.append(self.point(idx))
            idx = self.parent[idx]
        out.reverse()
        return out

    def edges(self):
        """All (parent_point, child_point) pairs, for drawing the tree."""
        return [(self.point(self.parent[i]), self.point(i))
                for i in range(self.n) if self.parent[i] >= 0]

    def __len__(self):
        return self.n

    def __repr__(self):
        return "Tree(%d nodes)" % self.n
