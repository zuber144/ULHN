"""
environment/rewards.py
======================
Configurable reward function for the ULHN Gymnasium environment.

All coefficients are taken from the config dict so they can be tuned in
config.yaml without touching source code.

Reward components:
    goal_reward           – large positive on reaching the goal
    progress_reward       – shaped reward based on distance improvement
    collision_penalty     – large negative on collision
    time_penalty          – small per-step penalty to encourage speed
    smoothness_penalty    – penalty for large velocity changes (jerk)
    obstacle_penalty      – penalty for being too close to obstacles
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RewardConfig:
    """All reward hyperparameters in one place."""

    goal_reward: float = 100.0
    collision_penalty: float = -50.0
    progress_scale: float = 10.0
    time_penalty: float = -0.1
    smoothness_scale: float = 0.5
    obstacle_clearance_penalty: float = -0.5
    obstacle_clearance_threshold: float = 1.0   # metres


class RewardFunction:
    """
    Stateful reward calculator.

    Stateful because it tracks the previous distance-to-goal to compute
    the shaping (progress) term.  Call ``reset()`` at the start of each
    episode.
    """

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()
        self._prev_distance: float = 0.0

    def reset(self, initial_distance: float) -> None:
        """Call at episode start with the initial robot-to-goal distance."""
        self._prev_distance = initial_distance

    def compute(
        self,
        *,
        current_distance: float,
        min_obstacle_distance: float,
        prev_v: float,
        prev_omega: float,
        current_v: float,
        current_omega: float,
        collision: bool,
        goal_reached: bool,
    ) -> float:
        """
        Compute the step reward.

        Parameters
        ----------
        current_distance:
            Current distance from robot to goal (m).
        min_obstacle_distance:
            Distance to the nearest obstacle boundary (m).
        prev_v, prev_omega:
            Velocities at the previous timestep.
        current_v, current_omega:
            Velocities at the current timestep.
        collision:
            True if a collision occurred this step.
        goal_reached:
            True if the goal was reached this step.

        Returns
        -------
        float : step reward
        """
        cfg = self.config
        reward = 0.0

        # --- Terminal events ---
        if goal_reached:
            reward += cfg.goal_reward
            self._prev_distance = current_distance
            return reward

        if collision:
            reward += cfg.collision_penalty
            self._prev_distance = current_distance
            return reward

        # --- Shaped progress reward ---
        distance_improvement = self._prev_distance - current_distance
        reward += cfg.progress_scale * distance_improvement

        # --- Per-step time penalty ---
        reward += cfg.time_penalty

        # --- Smoothness penalty (penalise large velocity changes) ---
        delta_v = abs(current_v - prev_v)
        delta_omega = abs(current_omega - prev_omega)
        reward -= cfg.smoothness_scale * (delta_v + 0.5 * delta_omega)

        # --- Obstacle proximity penalty ---
        if min_obstacle_distance < cfg.obstacle_clearance_threshold:
            reward += cfg.obstacle_clearance_penalty * (
                1.0 - min_obstacle_distance / cfg.obstacle_clearance_threshold
            )

        # Update distance for next step
        self._prev_distance = current_distance
        return float(reward)

    @classmethod
    def from_dict(cls, d: Dict) -> "RewardFunction":
        """Construct from a config dict (e.g. from config.yaml)."""
        cfg = RewardConfig(
            goal_reward=d.get("goal_reward", 100.0),
            collision_penalty=d.get("collision_penalty", -50.0),
            progress_scale=d.get("progress_scale", 10.0),
            time_penalty=d.get("time_penalty", -0.1),
            smoothness_scale=d.get("smoothness_scale", 0.5),
            obstacle_clearance_penalty=d.get("obstacle_clearance_penalty", -0.5),
            obstacle_clearance_threshold=d.get("obstacle_clearance_threshold", 1.0),
        )
        return cls(cfg)
