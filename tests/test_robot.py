"""tests/test_robot.py — Unit tests for the Robot kinematic model."""

from __future__ import annotations

import math
import pytest
import numpy as np
from environment.robot import Robot, RobotState


class TestRobotState:
    def test_position_tuple(self):
        s = RobotState(x=1.0, y=2.0)
        assert s.position() == (1.0, 2.0)

    def test_as_array_shape(self):
        s = RobotState(x=1.0, y=2.0, theta=0.5, v=0.3, omega=0.1)
        arr = s.as_array()
        assert arr.shape == (5,)
        assert arr.dtype == np.float32


class TestRobot:
    def setup_method(self):
        self.robot = Robot(radius=0.3, max_linear_velocity=1.0,
                           max_angular_velocity=1.5)

    def test_initial_state_after_reset(self):
        self.robot.reset(x=5.0, y=3.0, theta=0.0)
        assert self.robot.x == pytest.approx(5.0)
        assert self.robot.y == pytest.approx(3.0)
        assert self.robot.theta == pytest.approx(0.0)
        assert self.robot.v == 0.0
        assert self.robot.omega == 0.0

    def test_straight_line_motion(self):
        """Robot moving straight at v=1 for 1 second should travel 1 m in x."""
        self.robot.reset(x=0.0, y=0.0, theta=0.0)
        dt = 0.1
        for _ in range(10):
            self.robot.step(v_cmd=1.0, omega_cmd=0.0, dt=dt)
        assert self.robot.x == pytest.approx(1.0, abs=1e-6)
        assert self.robot.y == pytest.approx(0.0, abs=1e-6)

    def test_theta_wrapping(self):
        """Heading should remain in (-pi, pi]."""
        self.robot.reset(x=0.0, y=0.0, theta=3.0)
        self.robot.step(v_cmd=0.0, omega_cmd=1.0, dt=1.0)
        assert -math.pi < self.robot.theta <= math.pi

    def test_velocity_clipping(self):
        """Commands beyond physical limits should be clipped."""
        self.robot.reset(0.0, 0.0, 0.0)
        state = self.robot.step(v_cmd=999.0, omega_cmd=999.0, dt=0.1)
        assert abs(state.v) <= self.robot.max_linear_velocity + 1e-9
        assert abs(state.omega) <= self.robot.max_angular_velocity + 1e-9

    def test_distance_to(self):
        self.robot.reset(0.0, 0.0, 0.0)
        assert self.robot.distance_to(3.0, 4.0) == pytest.approx(5.0)

    def test_angle_to_ahead(self):
        """Target directly ahead should give angle ≈ 0."""
        self.robot.reset(0.0, 0.0, theta=0.0)
        angle = self.robot.angle_to(1.0, 0.0)
        assert angle == pytest.approx(0.0, abs=1e-9)

    def test_angle_to_behind(self):
        """Target directly behind should give angle ≈ ±pi."""
        self.robot.reset(0.0, 0.0, theta=0.0)
        angle = self.robot.angle_to(-1.0, 0.0)
        assert abs(angle) == pytest.approx(math.pi, abs=1e-9)

    def test_repr_contains_position(self):
        self.robot.reset(1.5, 2.5, 0.0)
        r = repr(self.robot)
        assert "1.50" in r
        assert "2.50" in r
