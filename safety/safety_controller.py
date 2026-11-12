"""
safety/safety_controller.py
============================
Top-level deterministic safety controller.

Receives:
    - PPO action (v_cmd, omega_cmd)
    - LiDAR observation
    - Anomaly score + threshold

Returns:
    - Final action (v_cmd, omega_cmd)
    - Safety state
    - Whether OOD was detected this step
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from environment.lidar import select_forward_sector
from safety.recovery import RecoveryStateMachine, RecoveryState
from safety.safety_rules import SafetyConfig, min_lidar_distance

logger = logging.getLogger(__name__)


@dataclass
class SafetyDecision:
    """Output of a single safety controller evaluation."""
    v_cmd: float
    omega_cmd: float
    state: RecoveryState
    is_ood: bool
    anomaly_score: float
    threshold: float
    min_obstacle_distance: float
    ppo_overridden: bool


class SafetyController:
    """
    Deterministic safety layer that gates PPO actions.

    Logic:
        IF anomaly_score < threshold AND min_dist > critical:
            → pass through PPO action unchanged
        ELSE:
            → consult recovery FSM for modified action
    """

    def __init__(
        self,
        config: SafetyConfig,
        threshold: float,
        lidar_max_range: float = 10.0,
        forward_sector_fraction: float = 0.5,
    ) -> None:
        self.config = config
        self.threshold = threshold
        self.lidar_max_range = lidar_max_range
        self.forward_sector_fraction = forward_sector_fraction

        self._fsm = RecoveryStateMachine(
            config=config,
            lidar_max_range=lidar_max_range,
        )
        self._emergency_stop_count = 0
        self._ood_detected_count = 0
        self._step_count = 0
        self._ood_consecutive_count = 0

    def reset(self) -> None:
        """Reset at episode start."""
        self._fsm.reset()
        self._emergency_stop_count = 0
        self._ood_detected_count = 0
        self._step_count = 0
        self._ood_consecutive_count = 0

    def step(
        self,
        ppo_v: float,
        ppo_omega: float,
        lidar_obs: np.ndarray,
        anomaly_score: float,
    ) -> SafetyDecision:
        """
        Evaluate the safety controller for one timestep.

        Parameters
        ----------
        ppo_v, ppo_omega:
            PPO-suggested velocity commands (physical units).
        lidar_obs:
            Normalised LiDAR observation.
        anomaly_score:
            Current anomaly score from the watchdog.

        Returns
        -------
        SafetyDecision
        """
        self._step_count += 1

        if anomaly_score > self.threshold:
            self._ood_consecutive_count += 1
        else:
            self._ood_consecutive_count = 0
        confirmation_steps = max(1, self.config.ood_confirmation_steps)
        is_ood = self._ood_consecutive_count >= confirmation_steps
        safety_lidar = select_forward_sector(
            lidar_obs, self.forward_sector_fraction
        )
        min_dist = min_lidar_distance(safety_lidar, self.lidar_max_range)

        if is_ood:
            self._ood_detected_count += 1

        # Update FSM
        new_state = self._fsm.update(
            is_ood=is_ood,
            anomaly_score=anomaly_score,
            threshold=self.threshold,
            lidar_obs=safety_lidar,
        )

        # Get modified action from FSM
        final_v, final_omega = self._fsm.modify_action(ppo_v, ppo_omega)

        ppo_overridden = (new_state != RecoveryState.NORMAL)
        if new_state == RecoveryState.STOP:
            self._emergency_stop_count += 1

        return SafetyDecision(
            v_cmd=final_v,
            omega_cmd=final_omega,
            state=new_state,
            is_ood=is_ood,
            anomaly_score=anomaly_score,
            threshold=self.threshold,
            min_obstacle_distance=min_dist,
            ppo_overridden=ppo_overridden,
        )

    @property
    def state(self) -> RecoveryState:
        return self._fsm.state

    @property
    def emergency_stop_count(self) -> int:
        return self._emergency_stop_count

    @property
    def ood_detected_count(self) -> int:
        return self._ood_detected_count

    @property
    def step_count(self) -> int:
        return self._step_count
