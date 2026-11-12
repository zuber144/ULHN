"""
environment/ulhn_env.py
=======================
Gymnasium-compatible environment for the ULHN project.

Observation space (Phase 1 — no LiDAR yet; LiDAR is added in Phase 2/3):

    [goal_distance (normalised), goal_angle (normalised), v, omega]

Action space:
    [linear_velocity, angular_velocity]  — continuous, ∈ [-1, 1]

The environment is seeded, reproducible, and configurable via config.yaml.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.robot import Robot
from environment.obstacles import ObstacleMap
from environment.physics import PhysicsEngine
from environment.lidar import LiDAR, LiDARConfig
from environment.rewards import RewardFunction, RewardConfig
from environment.scenarios import ScenarioGenerator, ScenarioConfig

logger = logging.getLogger(__name__)


class ULHNEnv(gym.Env):
    """
    Custom Gymnasium environment for ULHN.

    Observation vector layout::

        [ lidar_0, ..., lidar_{N-1},   # N normalised LiDAR ranges
          goal_dist,                    # normalised ∈ [0, 1]
          goal_angle,                   # normalised ∈ [-1, 1]
          v,                            # normalised ∈ [-1, 1]
          omega ]                       # normalised ∈ [-1, 1]

    Action vector::

        [ v_cmd, omega_cmd ]  ∈ [-1, 1]  (scaled to robot limits internally)

    Parameters
    ----------
    config : dict
        Full config dict loaded from config.yaml.
    render_mode : str or None
        'human' for Matplotlib visualisation, None for headless.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        cfg = config or {}
        env_cfg = cfg.get("environment", {})
        robot_cfg = cfg.get("robot", {})
        lidar_cfg = cfg.get("lidar", {})
        reward_cfg = cfg.get("rewards", {})
        obs_cfg = cfg.get("obstacles", {})
        goal_cfg = cfg.get("goal", {})

        # --- World parameters ---
        self.world_width: float = env_cfg.get("width", 20.0)
        self.world_height: float = env_cfg.get("height", 20.0)
        self.dt: float = env_cfg.get("timestep", 0.1)
        self.max_episode_steps: int = env_cfg.get("max_episode_steps", 500)
        self._base_seed: int = env_cfg.get("seed", 42)

        # --- Robot ---
        self.robot = Robot(
            radius=robot_cfg.get("radius", 0.3),
            max_linear_velocity=robot_cfg.get("max_linear_velocity", 1.0),
            max_angular_velocity=robot_cfg.get("max_angular_velocity", 1.5),
        )

        # --- LiDAR ---
        lidar_config = LiDARConfig(
            num_rays=lidar_cfg.get("num_rays", 36),
            fov_deg=lidar_cfg.get("fov_deg", 360.0),
            max_range=lidar_cfg.get("max_range", 10.0),
            noise_std=lidar_cfg.get("noise_std", 0.01),
            noise_enabled=lidar_cfg.get("noise_enabled", False),
        )
        self.lidar = LiDAR(lidar_config)

        # --- Reward ---
        self.reward_fn = RewardFunction.from_dict(reward_cfg)

        # --- Goal ---
        self.goal_radius: float = goal_cfg.get("radius", 0.5)

        # --- Scenario generator ---
        scenario_cfg = ScenarioConfig(
            world_width=self.world_width,
            world_height=self.world_height,
            num_static_min=obs_cfg.get("num_static_min", 5),
            num_static_max=obs_cfg.get("num_static_max", 12),
            rect_width_range=tuple(obs_cfg.get("rect_width_range", [0.5, 3.0])),
            rect_height_range=tuple(obs_cfg.get("rect_height_range", [0.5, 3.0])),
            circle_radius_range=tuple(obs_cfg.get("circle_radius_range", [0.3, 1.5])),
            rect_fraction=obs_cfg.get("rect_fraction", 0.6),
            margin=obs_cfg.get("margin", 0.5),
            robot_radius=self.robot.radius,
            goal_radius=self.goal_radius,
            min_goal_distance=goal_cfg.get("min_distance", 3.0),
        )
        self.scenario_gen = ScenarioGenerator(scenario_cfg, seed=self._base_seed)

        # Mutable scenario state
        self._obstacle_map: Optional[ObstacleMap] = None
        self._physics: Optional[PhysicsEngine] = None
        self._goal_x: float = 0.0
        self._goal_y: float = 0.0
        self._step_count: int = 0
        self._episode_seed: int = self._base_seed
        self._episode_count: int = 0

        # Velocity tracking for reward smoothness
        self._prev_v: float = 0.0
        self._prev_omega: float = 0.0

        # --- Observation & action spaces ---
        n_lidar = lidar_config.num_rays
        n_obs = n_lidar + 4   # lidar + goal_dist + goal_angle + v + omega

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_obs,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # --- Rendering ---
        self.render_mode = render_mode
        self._renderer: Optional[Any] = None  # Lazy import to avoid GUI on headless

        # Trajectory history (populated during episodes for visualisation)
        self.trajectory: list = []

        logger.info(
            "ULHNEnv initialised: world=%.0f×%.0f, lidar_rays=%d, obs_dim=%d",
            self.world_width, self.world_height, n_lidar, n_obs,
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to a new random scenario."""
        super().reset(seed=seed)

        # Derive per-episode seed
        if seed is not None:
            self._episode_seed = seed
        else:
            # Increment base seed so each episode is different but reproducible
            self._episode_seed = self._base_seed + self._episode_count

        self._episode_count += 1
        self.scenario_gen.reseed(self._episode_seed)

        # Generate new scenario
        (
            self._obstacle_map,
            start_x, start_y, start_theta,
            self._goal_x, self._goal_y,
        ) = self.scenario_gen.generate()

        # Place robot
        self.robot.reset(start_x, start_y, start_theta)

        # (Re)create physics engine
        self._physics = PhysicsEngine(
            robot=self.robot,
            obstacle_map=self._obstacle_map,
            dt=self.dt,
            goal_radius=self.goal_radius,
        )

        # Reset state
        self._step_count = 0
        self._prev_v = 0.0
        self._prev_omega = 0.0
        self.trajectory = [(start_x, start_y)]

        # Initialise reward function with starting distance
        initial_dist = self.robot.distance_to(self._goal_x, self._goal_y)
        self.reward_fn.reset(initial_dist)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply action and advance one timestep.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if self._physics is None:
            raise RuntimeError("Environment not reset.  Call env.reset() first.")

        # Scale normalised action [-1,1] → robot velocity limits
        v_cmd = float(action[0]) * self.robot.max_linear_velocity
        omega_cmd = float(action[1]) * self.robot.max_angular_velocity

        prev_v = self._prev_v
        prev_omega = self._prev_omega

        new_state, collision, goal_reached = self._physics.step(
            v_cmd, omega_cmd, self._goal_x, self._goal_y
        )

        self._step_count += 1
        self._prev_v = new_state.v
        self._prev_omega = new_state.omega
        self.trajectory.append((new_state.x, new_state.y))

        # Reward
        current_dist = self.robot.distance_to(self._goal_x, self._goal_y)
        min_obs_dist = self._physics.min_obstacle_distance()

        reward = self.reward_fn.compute(
            current_distance=current_dist,
            min_obstacle_distance=min_obs_dist,
            prev_v=prev_v,
            prev_omega=prev_omega,
            current_v=new_state.v,
            current_omega=new_state.omega,
            collision=collision,
            goal_reached=goal_reached,
        )

        terminated = collision or goal_reached
        truncated = self._step_count >= self.max_episode_steps

        obs = self._get_observation()
        info = self._get_info(
            collision=collision,
            goal_reached=goal_reached,
            min_obstacle_distance=min_obs_dist,
        )

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None
        renderer = self._get_renderer()
        return renderer.render(self)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Build the observation vector."""
        # LiDAR scan (already normalised to [0, 1])
        lidar_obs = self.lidar.scan(
            self.robot.x,
            self.robot.y,
            self.robot.theta,
            self._obstacle_map,
        )

        # Goal features (normalised)
        goal_dist = self.robot.distance_to(self._goal_x, self._goal_y)
        max_dist = math.hypot(self.world_width, self.world_height)
        goal_dist_norm = float(np.clip(goal_dist / max_dist, 0.0, 1.0))

        goal_angle = self.robot.angle_to(self._goal_x, self._goal_y)
        goal_angle_norm = float(goal_angle / math.pi)   # ∈ (-1, 1]

        # Velocity (normalised to [-1, 1])
        v_norm = self.robot.v / self.robot.max_linear_velocity
        omega_norm = self.robot.omega / self.robot.max_angular_velocity

        obs = np.concatenate([
            lidar_obs,
            [goal_dist_norm, goal_angle_norm, v_norm, omega_norm],
        ]).astype(np.float32)

        return obs

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _get_info(
        self,
        collision: bool = False,
        goal_reached: bool = False,
        min_obstacle_distance: float = float("inf"),
    ) -> Dict[str, Any]:
        return {
            "x": self.robot.x,
            "y": self.robot.y,
            "theta": self.robot.theta,
            "goal_x": self._goal_x,
            "goal_y": self._goal_y,
            "step": self._step_count,
            "collision": collision,
            "goal_reached": goal_reached,
            "min_obstacle_distance": min_obstacle_distance,
            "goal_distance": self.robot.distance_to(self._goal_x, self._goal_y),
            "episode_seed": self._episode_seed,
        }

    # ------------------------------------------------------------------
    # Lazy renderer
    # ------------------------------------------------------------------

    def _get_renderer(self):
        if self._renderer is None:
            from visualization.simulator_view import SimulatorRenderer
            self._renderer = SimulatorRenderer()
        return self._renderer

    # ------------------------------------------------------------------
    # Convenience properties (used by tests and dashboard)
    # ------------------------------------------------------------------

    @property
    def obstacle_map(self) -> Optional[ObstacleMap]:
        return self._obstacle_map

    @property
    def goal_position(self) -> Tuple[float, float]:
        return (self._goal_x, self._goal_y)

    @property
    def step_count(self) -> int:
        return self._step_count
