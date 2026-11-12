"""
environment/robot.py
====================
Differential-drive robot model.

State:
    x       – X position (m)
    y       – Y position (m)
    theta   – Heading angle (rad)
    v       – Linear velocity (m/s)
    omega   – Angular velocity (rad/s)

Kinematics (Euler integration):
    x(t+1)     = x(t)     + v * cos(theta) * dt
    y(t+1)     = y(t)     + v * sin(theta) * dt
    theta(t+1) = theta(t) + omega * dt
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class RobotState:
    """Immutable snapshot of robot state at a given timestep."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0        # heading in radians
    v: float = 0.0            # linear velocity  (m/s)
    omega: float = 0.0        # angular velocity (rad/s)

    def position(self) -> Tuple[float, float]:
        """Return (x, y) position tuple."""
        return (self.x, self.y)

    def as_array(self) -> np.ndarray:
        """Return state as a numpy array [x, y, theta, v, omega]."""
        return np.array([self.x, self.y, self.theta, self.v, self.omega],
                        dtype=np.float32)


class Robot:
    """
    Differential-drive robot with configurable limits.

    The robot stores its own mutable state and exposes a `step()` method
    that integrates the kinematic model forward by one timestep.
    """

    def __init__(
        self,
        radius: float = 0.3,
        max_linear_velocity: float = 1.0,
        max_angular_velocity: float = 1.5,
    ) -> None:
        """
        Parameters
        ----------
        radius:
            Body radius in metres (used for collision detection).
        max_linear_velocity:
            Maximum allowed |v| in m/s.
        max_angular_velocity:
            Maximum allowed |omega| in rad/s.
        """
        self.radius = radius
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

        self._state = RobotState()

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> RobotState:
        """Current robot state (read-only snapshot)."""
        return RobotState(
            x=self._state.x,
            y=self._state.y,
            theta=self._state.theta,
            v=self._state.v,
            omega=self._state.omega,
        )

    @property
    def x(self) -> float:
        return self._state.x

    @property
    def y(self) -> float:
        return self._state.y

    @property
    def theta(self) -> float:
        return self._state.theta

    @property
    def v(self) -> float:
        return self._state.v

    @property
    def omega(self) -> float:
        return self._state.omega

    def position(self) -> Tuple[float, float]:
        return (self._state.x, self._state.y)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def reset(self, x: float, y: float, theta: float) -> None:
        """
        Reset robot to a specified pose with zero velocities.

        Parameters
        ----------
        x, y:
            Initial position (m).
        theta:
            Initial heading (rad).
        """
        self._state = RobotState(x=x, y=y, theta=theta, v=0.0, omega=0.0)

    def step(self, v_cmd: float, omega_cmd: float, dt: float) -> RobotState:
        """
        Apply velocity commands and integrate kinematics.

        The commands are clipped to the robot's physical limits before
        integration.

        Parameters
        ----------
        v_cmd:
            Desired linear velocity (m/s).  Clipped to
            [-max_linear_velocity, max_linear_velocity].
        omega_cmd:
            Desired angular velocity (rad/s). Clipped to
            [-max_angular_velocity, max_angular_velocity].
        dt:
            Timestep (s).

        Returns
        -------
        New RobotState after integration.
        """
        # Clip to physical limits
        v = float(np.clip(v_cmd, -self.max_linear_velocity,
                          self.max_linear_velocity))
        omega = float(np.clip(omega_cmd, -self.max_angular_velocity,
                              self.max_angular_velocity))

        # Euler integration of unicycle / differential-drive kinematics
        theta_new = self._state.theta + omega * dt
        x_new = self._state.x + v * math.cos(self._state.theta) * dt
        y_new = self._state.y + v * math.sin(self._state.theta) * dt

        # Normalise theta to (-pi, pi]
        theta_new = math.atan2(math.sin(theta_new), math.cos(theta_new))

        self._state = RobotState(
            x=x_new,
            y=y_new,
            theta=theta_new,
            v=v,
            omega=omega,
        )
        return self.state

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def distance_to(self, x: float, y: float) -> float:
        """Euclidean distance from robot centre to point (x, y)."""
        return math.hypot(x - self._state.x, y - self._state.y)

    def angle_to(self, x: float, y: float) -> float:
        """
        Bearing from robot to point (x, y), relative to robot heading.
        Returns a value in (-pi, pi].
        """
        absolute_angle = math.atan2(y - self._state.y, x - self._state.x)
        relative_angle = absolute_angle - self._state.theta
        return math.atan2(math.sin(relative_angle), math.cos(relative_angle))

    def __repr__(self) -> str:
        s = self._state
        return (
            f"Robot(x={s.x:.2f}, y={s.y:.2f}, theta={math.degrees(s.theta):.1f}°, "
            f"v={s.v:.2f}, ω={s.omega:.2f})"
        )
