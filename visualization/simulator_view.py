"""
visualization/simulator_view.py
================================
Matplotlib-based renderer for the ULHN 2D simulator.

Renders:
    - World boundary
    - Static obstacles (rectangles and circles)
    - Robot body + heading arrow
    - Goal marker
    - Robot trajectory (path history)
    - LiDAR rays (optional)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

if TYPE_CHECKING:
    from environment.ulhn_env import ULHNEnv
    from environment.obstacles import RectObstacle, CircleObstacle

# Use TkAgg or Agg depending on whether a display is available
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")


class SimulatorRenderer:
    """
    Matplotlib renderer for the ULHN environment.

    Supports both interactive ('human') and off-screen ('rgb_array') modes.
    """

    def __init__(self, figsize: tuple = (8, 8), show_lidar: bool = True) -> None:
        self.figsize = figsize
        self.show_lidar = show_lidar
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None

    def _init_figure(self, env: "ULHNEnv") -> None:
        """Lazily create the Matplotlib figure."""
        if self._fig is not None:
            return
        self._fig, self._ax = plt.subplots(figsize=self.figsize)
        self._fig.tight_layout()
        plt.ion()
        plt.show(block=False)

    def render(self, env: "ULHNEnv") -> Optional[np.ndarray]:
        """
        Render the current environment state.

        Returns an RGBA numpy array if render_mode is 'rgb_array'.
        """
        self._init_figure(env)
        ax = self._ax
        ax.clear()

        w = env.world_width
        h = env.world_height

        # --- World boundary ---
        ax.set_xlim(-0.5, w + 0.5)
        ax.set_ylim(-0.5, h + 0.5)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        self._fig.patch.set_facecolor("#0f0f1e")
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_color("#444466")

        world_rect = patches.Rectangle(
            (0, 0), w, h,
            linewidth=2, edgecolor="#4488ff", facecolor="none", linestyle="--",
        )
        ax.add_patch(world_rect)

        # --- Obstacles ---
        if env.obstacle_map is not None:
            for obs in env.obstacle_map.obstacles:
                self._draw_obstacle(ax, obs)

        # --- Trajectory ---
        if len(env.trajectory) > 1:
            xs = [p[0] for p in env.trajectory]
            ys = [p[1] for p in env.trajectory]
            ax.plot(xs, ys, color="#44aaff", linewidth=1.2,
                    alpha=0.6, zorder=3, label="Path")

        # --- LiDAR rays ---
        if self.show_lidar and env.obstacle_map is not None:
            self._draw_lidar(ax, env)

        # --- Goal ---
        gx, gy = env.goal_position
        goal_circle = plt.Circle(
            (gx, gy), env.goal_radius,
            color="#00ff88", alpha=0.4, zorder=5,
        )
        ax.add_patch(goal_circle)
        ax.plot(gx, gy, marker="*", markersize=14,
                color="#00ff88", zorder=6, label="Goal")

        # --- Robot ---
        self._draw_robot(ax, env)

        # --- Labels ---
        ax.set_title(
            f"ULHN Simulator  |  Step {env.step_count}",
            color="#ccccff", fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("X (m)", color="#aaaaaa")
        ax.set_ylabel("Y (m)", color="#aaaaaa")

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right",
                      facecolor="#222244", edgecolor="#4444aa",
                      labelcolor="#ccccff", fontsize=9)

        if env.render_mode == "rgb_array":
            self._fig.canvas.draw()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return buf

        plt.pause(0.001)
        return None

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_obstacle(self, ax: plt.Axes, obs) -> None:
        from environment.obstacles import RectObstacle, CircleObstacle

        if isinstance(obs, RectObstacle):
            rect = patches.Rectangle(
                (obs.cx - obs.half_w, obs.cy - obs.half_h),
                2 * obs.half_w,
                2 * obs.half_h,
                linewidth=1.5,
                edgecolor="#ff6644",
                facecolor="#3d1a0e",
                zorder=4,
            )
            ax.add_patch(rect)

        elif isinstance(obs, CircleObstacle):
            circle = plt.Circle(
                (obs.cx, obs.cy),
                obs.radius,
                color="#3d1a0e",
                linewidth=1.5,
                edgecolor="#ff6644",
                zorder=4,
            )
            ax.add_patch(circle)

    def _draw_robot(self, ax: plt.Axes, env: "ULHNEnv") -> None:
        rx, ry, rtheta = env.robot.x, env.robot.y, env.robot.theta
        r = env.robot.radius

        # Body
        body = plt.Circle((rx, ry), r, color="#4488ff",
                           alpha=0.85, zorder=7)
        ax.add_patch(body)

        # Heading arrow
        arrow_len = r * 1.8
        ax.annotate(
            "",
            xy=(rx + arrow_len * math.cos(rtheta),
                ry + arrow_len * math.sin(rtheta)),
            xytext=(rx, ry),
            arrowprops=dict(arrowstyle="->", color="#ffffff",
                            lw=2.0),
            zorder=8,
        )

        # Start marker (first trajectory point)
        if env.trajectory:
            sx, sy = env.trajectory[0]
            ax.plot(sx, sy, marker="o", markersize=8,
                    color="#ffaa00", zorder=6, label="Start")

    def _draw_lidar(self, ax: plt.Axes, env: "ULHNEnv") -> None:
        """Draw LiDAR rays as thin lines."""
        lidar = env.lidar
        rx, ry, rtheta = env.robot.x, env.robot.y, env.robot.theta
        fov_rad = math.radians(lidar.config.fov_deg)

        if lidar.config.num_rays == 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(
                -fov_rad / 2.0,
                fov_rad / 2.0,
                lidar.config.num_rays,
                endpoint=(lidar.config.fov_deg < 360.0),
            ).tolist()

        for offset in offsets:
            angle = rtheta + offset
            dist = env.obstacle_map.cast_ray(
                rx, ry, angle, lidar.config.max_range
            )
            end_x = rx + dist * math.cos(angle)
            end_y = ry + dist * math.sin(angle)
            ax.plot([rx, end_x], [ry, end_y],
                    color="#00ffff", alpha=0.15, linewidth=0.5, zorder=2)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
