"""
Drawing the arm: the world view and, for a 2-joint arm, joint space itself.

The two-panel figure is the classic way to explain configuration space:

  left   the arm in the world, drawn at several poses along its path
  right  joint space: every pixel is one (q1, q2) pair, dark where that pose
         collides. The planner's tree and path live in THIS picture. A
         straight line on the right is a smooth turn of both joints on the
         left; obstacles that are simple shapes on the left become strange
         curved blobs on the right.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

from env.obstacles import CircleObstacle, RectObstacle
from viz.charts import INK_PRIMARY, INK_SECONDARY, SURFACE
from viz.plot import GOAL_COLOR, OBSTACLE_EDGE, OBSTACLE_FACE, PLANNER_COLORS, START_COLOR

from .environment import JOINT_LIMIT

ARM_COLOR = PLANNER_COLORS["RRT*"]
# Sequential ramp (one hue, light -> dark) for "progress along the path".
PROGRESS = LinearSegmentedColormap.from_list("progress", ["#b8e6d3", ARM_COLOR, "#0b5e40"])


def draw_world(ax, env):
    for o in env.obstacles:
        if isinstance(o, RectObstacle):
            ax.add_patch(Rectangle((o.x, o.y), o.w, o.h, facecolor=OBSTACLE_FACE,
                                   edgecolor=OBSTACLE_EDGE, linewidth=0.8, zorder=2))
        elif isinstance(o, CircleObstacle):
            ax.add_patch(Circle((o.cx, o.cy), o.r, facecolor=OBSTACLE_FACE,
                                edgecolor=OBSTACLE_EDGE, linewidth=0.8, zorder=2))
    ax.add_patch(Circle(tuple(env.arm.base), env.arm.reach, fill=False, linestyle=":",
                        edgecolor="#c3c2b7", linewidth=1.0, zorder=1))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#a0aec0")


def draw_arm(ax, env, q, color, alpha=1.0, linewidth=None, zorder=4):
    pts = env.arm.joint_positions(q)
    lw = linewidth if linewidth is not None else 2 * env.arm.link_radius * 2.2
    ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=alpha, linewidth=lw,
            solid_capstyle="round", zorder=zorder)
    ax.plot(pts[:-1, 0], pts[:-1, 1], "o", color="white", markersize=2.2,
            alpha=alpha, zorder=zorder + 0.1)


def interpolate_path(path, n_poses):
    """Evenly spaced poses along a joint-space path (by joint-space length)."""
    P = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    targets = np.linspace(0, s[-1], n_poses)
    out = []
    for t in targets:
        k = min(np.searchsorted(s, t, side="right") - 1, len(seg) - 1)
        f = 0.0 if seg[k] == 0 else (t - s[k]) / seg[k]
        out.append(P[k] + f * (P[k + 1] - P[k]))
    return out


def draw_path_poses(ax, env, path, n_poses=9):
    poses = interpolate_path(path, n_poses)
    for k, q in enumerate(poses[1:-1], start=1):
        draw_arm(ax, env, q, PROGRESS(k / (n_poses - 1)), alpha=0.75, zorder=3)
    draw_arm(ax, env, path[0], START_COLOR, zorder=5)
    draw_arm(ax, env, path[-1], GOAL_COLOR, zorder=5)


def cspace_mask(env, resolution=240):
    """(resolution, resolution) bool image of a 2-joint arm's joint space; True = collides."""
    if env.arm.dof != 2:
        raise ValueError("joint space can only be drawn for 2 joints")
    g = np.linspace(-JOINT_LIMIT, JOINT_LIMIT, resolution)
    q1, q2 = np.meshgrid(g, g)
    Q = np.stack([q1.ravel(), q2.ravel()], axis=1)
    return env._collides(Q).reshape(resolution, resolution)


def figure_two_link(env, planner, path, title, out_path):
    fig = plt.figure(figsize=(11.5, 5.6), dpi=150, facecolor=SURFACE)
    fig.text(0.015, 0.97, title, fontsize=13, fontweight="bold", color=INK_PRIMARY, va="top")
    fig.text(0.015, 0.915,
             "Left: the arm in the world (green = start, red = goal, teal = poses along the path). "
             "Right: joint space. Each pixel is one pair of joint angles,\n"
             "dark where that pose hits something. The planner searches on the right; "
             "a straight line there is a smooth turn of both joints on the left.",
             fontsize=9, color=INK_SECONDARY, va="top", linespacing=1.4)

    ax1 = fig.add_axes([0.02, 0.05, 0.44, 0.76])
    draw_world(ax1, env)
    draw_path_poses(ax1, env, path)
    ax1.set_title("World", fontsize=10, loc="left", color=INK_PRIMARY)

    ax2 = fig.add_axes([0.54, 0.09, 0.42, 0.72])
    mask = cspace_mask(env)
    ax2.imshow(mask, origin="lower", extent=(-JOINT_LIMIT, JOINT_LIMIT, -JOINT_LIMIT, JOINT_LIMIT),
               cmap=ListedColormap([SURFACE, OBSTACLE_FACE]), interpolation="nearest", zorder=1)
    if planner.tree is not None:
        ax2.add_collection(LineCollection(planner.tree.edges(), colors=ARM_COLOR,
                                          linewidths=0.5, alpha=0.5, zorder=2))
    P = np.asarray(path)
    ax2.plot(P[:, 0], P[:, 1], color=INK_PRIMARY, linewidth=2.0, zorder=4)
    ax2.plot(*env.start, "o", color=START_COLOR, markersize=8, markeredgecolor="white", zorder=5)
    ax2.plot(*env.goal, "*", color=GOAL_COLOR, markersize=14, markeredgecolor="white", zorder=5)
    ticks = [-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi]
    labels = ["-π", "-π/2", "0", "π/2", "π"]
    ax2.set_xticks(ticks, labels)
    ax2.set_yticks(ticks, labels)
    ax2.set_xlabel("q1: shoulder angle (rad)", fontsize=9, color=INK_SECONDARY)
    ax2.set_ylabel("q2: elbow angle (rad)", fontsize=9, color=INK_SECONDARY)
    ax2.tick_params(labelsize=8, colors=INK_SECONDARY)
    ax2.set_title("Joint space (configuration space)", fontsize=10, loc="left", color=INK_PRIMARY)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def figure_arm_path(env, path, title, subtitle, out_path, n_poses=10):
    fig = plt.figure(figsize=(6.2, 6.8), dpi=150, facecolor=SURFACE)
    fig.text(0.03, 0.975, title, fontsize=12.5, fontweight="bold", color=INK_PRIMARY, va="top")
    fig.text(0.03, 0.935, subtitle, fontsize=9, color=INK_SECONDARY, va="top", linespacing=1.4)
    ax = fig.add_axes([0.04, 0.03, 0.92, 0.84])
    draw_world(ax, env)
    draw_path_poses(ax, env, path, n_poses)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path
