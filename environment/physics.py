"""
environment/physics.py
======================
Thin physics engine that ties together the robot kinematics and the obstacle
map for the ULHN 2D simulator.

Responsibilities:
    1. Step the robot forward in time.
    2. Check for collisions between the robot body and obstacles / walls.
    3. Enforce world boundary limits (optional: clamp vs. flag).
    4. Check goal-reaching condition.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

from environment.robot import Robot, RobotState
from environment.obstacles import ObstacleMap

logger = logging.getLogger(__name__)


class PhysicsEngine:
    """
    Integrates robot motion and detects collisions and goal events.

    The engine itself is stateless with respect to robot or obstacle data —
    those objects are passed in at construction and mutated in place.
    """

    def __init__(
        self,
        robot: Robot,
        obstacle_map: ObstacleMap,
        dt: float = 0.1,
        goal_radius: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        robot:
            The robot instance to integrate.
        obstacle_map:
            The world obstacle map.
        dt:
            Simulation timestep (s).
        goal_radius:
            Capture radius for the goal (m).  Goal is reached when the
            distance from the robot centre to the goal position falls
            below this value.
        """
        self.robot = robot
        self.obstacle_map = obstacle_map
        self.dt = dt
        self.goal_radius = goal_radius

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        v_cmd: float,
        omega_cmd: float,
        goal_x: float,
        goal_y: float,
    ) -> Tuple[RobotState, bool, bool]:
        """
        Advance simulation by one timestep.

        Parameters
        ----------
        v_cmd, omega_cmd:
            Velocity commands (clipped by Robot internally).
        goal_x, goal_y:
            Current goal position.

        Returns
        -------
        new_state : RobotState
            Robot state after integration.
        collision : bool
            True if robot collided with an obstacle or wall.
        goal_reached : bool
            True if robot is within *goal_radius* of the goal.
        """
        new_state = self.robot.step(v_cmd, omega_cmd, self.dt)

        collision = self._check_collision(new_state.x, new_state.y)
        goal_reached = self._check_goal(new_state.x, new_state.y,
                                        goal_x, goal_y)

        if collision:
            logger.debug(
                "Collision at (%.2f, %.2f)", new_state.x, new_state.y
            )

        return new_state, collision, goal_reached

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def _check_collision(self, x: float, y: float) -> bool:
        """
        Check whether the robot body (circle with radius = robot.radius)
        overlaps any obstacle or the world boundary.
        """
        return self.obstacle_map.point_in_obstacle(
            x, y, margin=self.robot.radius
        )

    # ------------------------------------------------------------------
    # Goal detection
    # ------------------------------------------------------------------

    def _check_goal(
        self,
        x: float,
        y: float,
        goal_x: float,
        goal_y: float,
    ) -> bool:
        """Return True if the robot has entered the goal capture radius."""
        return math.hypot(goal_x - x, goal_y - y) <= self.goal_radius

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def min_obstacle_distance(self) -> float:
        """
        Return the distance from the current robot position to the nearest
        obstacle boundary.
        """
        return self.obstacle_map.min_distance_to_obstacle(
            self.robot.x, self.robot.y
        )
