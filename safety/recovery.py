"""
safety/recovery.py
==================
Recovery state machine for the ULHN safety system.

States:
    NORMAL    – PPO is in control; no anomaly detected.
    SLOWDOWN  – Anomaly detected; reduce speed but continue.
    STOP      – Severe anomaly or critical obstacle proximity; halt.
    REORIENT  – Robot is stopped; perform slow rotation to find clear path.
    RESUME    – Anomaly cleared; counting down consecutive normal readings.

Transitions:
    NORMAL → SLOWDOWN   : anomaly score > threshold
    SLOWDOWN → STOP     : severe anomaly OR min_obs_dist < critical_distance
    STOP → REORIENT     : stopped for > stop_hold_steps
    REORIENT → RESUME   : anomaly score < threshold
    RESUME → NORMAL     : resume_consecutive normal readings accumulated
    RESUME → SLOWDOWN   : anomaly detected again during cooldown
    Any → STOP          : min_obs_dist < critical_distance (emergency override)
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Tuple

import numpy as np

from safety.safety_rules import SafetyConfig

logger = logging.getLogger(__name__)


class RecoveryState(Enum):
    NORMAL   = auto()
    SLOWDOWN = auto()
    STOP     = auto()
    REORIENT = auto()
    RESUME   = auto()


class RecoveryStateMachine:
    """
    Deterministic state machine governing robot recovery from OOD events.

    All logic is rule-based — no neural network is involved.
    """

    def __init__(
        self,
        config: SafetyConfig,
        lidar_max_range: float = 10.0,
        stop_hold_steps: int = 5,
    ) -> None:
        self.config = config
        self.lidar_max_range = lidar_max_range
        self.stop_hold_steps = stop_hold_steps  # steps to hold STOP before REORIENT

        self._state = RecoveryState.NORMAL
        self._consecutive_normal: int = 0
        self._stop_hold_counter: int = 0
        self._reorient_direction: float = 1.0  # +1 or -1 angular
        self._step_count: int = 0

    def reset(self) -> None:
        """Reset state machine to NORMAL at episode start."""
        self._state = RecoveryState.NORMAL
        self._consecutive_normal = 0
        self._stop_hold_counter = 0
        self._reorient_direction = 1.0
        self._step_count = 0

    @property
    def state(self) -> RecoveryState:
        return self._state

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        is_ood: bool,
        anomaly_score: float,
        threshold: float,
        lidar_obs: np.ndarray,
    ) -> RecoveryState:
        self._step_count += 1

        min_dist = float(np.min(lidar_obs)) * self.lidar_max_range

        # Emergency override: critical obstacle proximity
        # If stop_requires_ood=True, only stop when OOD also active (prevents
        # over-triggering during normal close-approach navigation)
        if (
            self._state not in (RecoveryState.STOP, RecoveryState.REORIENT)
            and min_dist < self.config.critical_distance
        ):
            if not self.config.stop_requires_ood or is_ood:
                self._transition(RecoveryState.STOP, "critical obstacle proximity")
                return self._state

        # State transitions
        if self._state == RecoveryState.NORMAL:
            if is_ood:
                self._consecutive_normal = 0
                self._transition(RecoveryState.SLOWDOWN, "anomaly detected")

        elif self._state == RecoveryState.SLOWDOWN:
            if not is_ood:
                self._consecutive_normal += 1
                if self._consecutive_normal >= self.config.resume_consecutive:
                    self._transition(RecoveryState.NORMAL, "anomaly cleared in SLOWDOWN")
                    self._consecutive_normal = 0
            else:
                self._consecutive_normal = 0
                # Severe anomaly → STOP
                if anomaly_score > threshold * 2.0:
                    self._transition(RecoveryState.STOP, "severe anomaly")

        elif self._state == RecoveryState.STOP:
            self._stop_hold_counter += 1
            if self._stop_hold_counter >= self.stop_hold_steps:
                self._stop_hold_counter = 0
                self._consecutive_normal = 0
                self._transition(RecoveryState.REORIENT, "held stop; starting reorientation")

        elif self._state == RecoveryState.REORIENT:
            if not is_ood:
                self._consecutive_normal += 1
                if self._consecutive_normal >= self.config.resume_consecutive:
                    self._consecutive_normal = 0
                    self._transition(RecoveryState.RESUME, "anomaly cleared; resuming")
            else:
                self._consecutive_normal = 0
                if np.random.random() < 0.05:
                    self._reorient_direction *= -1.0

        elif self._state == RecoveryState.RESUME:
            if not is_ood:
                self._consecutive_normal += 1
                if self._consecutive_normal >= self.config.resume_consecutive:
                    self._transition(RecoveryState.NORMAL, "resume complete")
                    self._consecutive_normal = 0
            else:
                self._consecutive_normal = 0
                self._transition(RecoveryState.SLOWDOWN, "re-anomaly during resume")

        return self._state

    def _transition(self, new_state: RecoveryState, reason: str) -> None:
        if new_state != self._state:
            logger.info(
                "Safety FSM: %s → %s  (%s)",
                self._state.name, new_state.name, reason,
            )
            self._state = new_state

    # ------------------------------------------------------------------
    # Action modifiers
    # ------------------------------------------------------------------

    def modify_action(
        self,
        ppo_v: float,
        ppo_omega: float,
    ) -> Tuple[float, float]:
        """
        Return (v_cmd, omega_cmd) adjusted for the current safety state.

        Parameters
        ----------
        ppo_v, ppo_omega:
            Raw PPO velocity commands (already in physical units).

        Returns
        -------
        (v_cmd, omega_cmd) : adjusted velocity commands.
        """
        cfg = self.config

        if self._state == RecoveryState.NORMAL:
            return ppo_v, ppo_omega

        elif self._state == RecoveryState.SLOWDOWN:
            return ppo_v * cfg.slowdown_factor, ppo_omega * cfg.slowdown_factor

        elif self._state == RecoveryState.STOP:
            return 0.0, 0.0

        elif self._state == RecoveryState.REORIENT:
            return 0.0, cfg.recovery_speed * self._reorient_direction

        elif self._state == RecoveryState.RESUME:
            return ppo_v * cfg.slowdown_factor, ppo_omega * cfg.slowdown_factor

        return ppo_v, ppo_omega
