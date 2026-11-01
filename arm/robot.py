"""
A planar robot arm: N straight links joined by N rotary joints.

    base --link 1-- joint 2 --link 2-- joint 3 ... --link N-- tip

The arm's *configuration* is the vector of joint angles q = (q1, ..., qN).
Each angle is measured relative to the previous link, so

    heading of link k  =  q1 + q2 + ... + qk

and forward kinematics (joint angles -> where the links are in the world) is
just a running sum of headings and link vectors.

This is the key shift from the 2-D mobile robot. There, a configuration *was*
a point in the world. Here a configuration is a point in "joint space" -- an
N-dimensional box of angles -- and every such point corresponds to the whole
arm occupying a set of line segments in the world. Planning happens in joint
space; collision checking happens in the world.
"""

import numpy as np


class PlanarArm:
    """
    Parameters
    ----------
    link_lengths : sequence of float
        Length of each link, base outwards. len() = number of joints (DOF).
    base : (x, y)
        Where the first joint is fixed in the world.
    link_radius : float
        Each link is a capsule (a thick line) of this radius, so collision
        checks keep this much clearance -- the arm analogue of the mobile
        robot's disc radius.
    """

    def __init__(self, link_lengths, base=(50.0, 50.0), link_radius=1.0):
        self.link_lengths = np.asarray(link_lengths, dtype=float)
        if self.link_lengths.ndim != 1 or len(self.link_lengths) < 1:
            raise ValueError("need at least one link")
        self.base = np.asarray(base, dtype=float)
        self.link_radius = float(link_radius)

        # reach_from[i] = total length from joint i to the tip. Used to bound
        # how far any point of the arm can move when the joints turn (see
        # ArmEnvironment.is_motion_valid).
        self.reach_from = np.cumsum(self.link_lengths[::-1])[::-1].copy()

    @property
    def dof(self):
        """Degrees of freedom = number of joints."""
        return len(self.link_lengths)

    @property
    def reach(self):
        return float(self.link_lengths.sum())

    def joint_positions(self, q):
        """World positions of the base, every joint and the tip: shape (N+1, 2)."""
        return self.joint_positions_batch(np.asarray(q, dtype=float)[None, :])[0]

    def joint_positions_batch(self, Q):
        """Forward kinematics for many configurations at once.

        Q : (M, N) joint angles  ->  (M, N+1, 2) points.
        Vectorised because motion checking evaluates dozens of poses per edge.
        """
        Q = np.asarray(Q, dtype=float)
        headings = np.cumsum(Q, axis=1)                          # (M, N)
        steps = np.stack([np.cos(headings), np.sin(headings)], axis=2)
        steps *= self.link_lengths[None, :, None]                # (M, N, 2)
        out = np.empty((Q.shape[0], self.dof + 1, 2))
        out[:, 0, :] = self.base
        out[:, 1:, :] = self.base + np.cumsum(steps, axis=1)
        return out

    def __repr__(self):
        return "PlanarArm(dof=%d, links=%s, base=%s)" % (
            self.dof, np.round(self.link_lengths, 2).tolist(), self.base.tolist())
