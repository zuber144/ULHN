"""tests/test_safety.py — Unit tests for Safety Controller and Recovery FSM."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _load(name, relpath):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("safety.safety_rules",     "safety/safety_rules.py")
_load("safety.recovery",         "safety/recovery.py")
_load("safety.safety_controller","safety/safety_controller.py")

from safety.safety_rules import SafetyConfig
from safety.recovery import RecoveryStateMachine, RecoveryState
from safety.safety_controller import SafetyController


CFG = SafetyConfig(
    critical_distance=0.5,
    slowdown_distance=1.5,
    slowdown_factor=0.5,
    recovery_speed=0.2,
    resume_consecutive=3,
)


class TestRecoveryFSM:
    def setup_method(self):
        self.fsm = RecoveryStateMachine(CFG, lidar_max_range=10.0, stop_hold_steps=2)

    def _lidar_clear(self) -> np.ndarray:
        """Normalised LiDAR with all rays at max range (no obstacles near)."""
        return np.ones(36, dtype=np.float32)

    def _lidar_close(self) -> np.ndarray:
        """Normalised LiDAR with one very close ray (critical distance)."""
        obs = np.ones(36, dtype=np.float32)
        obs[18] = 0.03  # 0.3 m — below critical_distance=0.5
        return obs

    def test_initial_state_normal(self):
        assert self.fsm.state == RecoveryState.NORMAL

    def test_ood_transitions_to_slowdown(self):
        self.fsm.update(is_ood=True, anomaly_score=0.1, threshold=0.05,
                        lidar_obs=self._lidar_clear())
        assert self.fsm.state == RecoveryState.SLOWDOWN

    def test_critical_distance_triggers_stop(self):
        self.fsm.update(is_ood=False, anomaly_score=0.0, threshold=0.05,
                        lidar_obs=self._lidar_close())
        assert self.fsm.state == RecoveryState.STOP

    def test_recovery_from_stop_to_reorient(self):
        # Get to STOP state
        self.fsm.update(is_ood=False, anomaly_score=0.0, threshold=0.05,
                        lidar_obs=self._lidar_close())
        assert self.fsm.state == RecoveryState.STOP
        # Hold enough steps
        for _ in range(self.fsm.stop_hold_steps):
            self.fsm.update(is_ood=False, anomaly_score=0.0, threshold=0.05,
                            lidar_obs=self._lidar_clear())
        assert self.fsm.state == RecoveryState.REORIENT

    def test_persistent_close_ood_does_not_restart_stop_reorient(self):
        config = SafetyConfig(
            critical_distance=0.5,
            stop_requires_ood=True,
            resume_consecutive=3,
        )
        fsm = RecoveryStateMachine(config, lidar_max_range=10.0, stop_hold_steps=2)

        fsm.update(is_ood=True, anomaly_score=0.1, threshold=0.05,
                   lidar_obs=self._lidar_close())
        assert fsm.state == RecoveryState.STOP

        for _ in range(fsm.stop_hold_steps):
            fsm.update(is_ood=True, anomaly_score=0.1, threshold=0.05,
                       lidar_obs=self._lidar_close())
        assert fsm.state == RecoveryState.REORIENT

        for _ in range(4):
            fsm.update(is_ood=True, anomaly_score=0.1, threshold=0.05,
                       lidar_obs=self._lidar_close())
            assert fsm.state == RecoveryState.REORIENT

    def test_reorient_waits_for_consecutive_normal_readings(self):
        config = SafetyConfig(
            critical_distance=0.5,
            stop_requires_ood=True,
            resume_consecutive=3,
        )
        fsm = RecoveryStateMachine(config, lidar_max_range=10.0, stop_hold_steps=1)
        fsm.update(is_ood=True, anomaly_score=0.1, threshold=0.05,
                   lidar_obs=self._lidar_close())
        fsm.update(is_ood=True, anomaly_score=0.1, threshold=0.05,
                   lidar_obs=self._lidar_close())
        assert fsm.state == RecoveryState.REORIENT

        fsm.update(is_ood=False, anomaly_score=0.0, threshold=0.05,
                   lidar_obs=self._lidar_clear())
        fsm.update(is_ood=True, anomaly_score=0.1, threshold=0.05,
                   lidar_obs=self._lidar_close())
        assert fsm.state == RecoveryState.REORIENT

        for _ in range(config.resume_consecutive - 1):
            fsm.update(is_ood=False, anomaly_score=0.0, threshold=0.05,
                       lidar_obs=self._lidar_clear())
            assert fsm.state == RecoveryState.REORIENT
        fsm.update(is_ood=False, anomaly_score=0.0, threshold=0.05,
                   lidar_obs=self._lidar_clear())
        assert fsm.state == RecoveryState.RESUME

    def test_normal_action_unchanged(self):
        v, omega = self.fsm.modify_action(0.8, 0.3)
        assert v == pytest.approx(0.8)
        assert omega == pytest.approx(0.3)

    def test_slowdown_action_scaled(self):
        self.fsm.update(is_ood=True, anomaly_score=0.1, threshold=0.05,
                        lidar_obs=self._lidar_clear())
        v, omega = self.fsm.modify_action(1.0, 1.0)
        assert v == pytest.approx(CFG.slowdown_factor)
        assert omega == pytest.approx(CFG.slowdown_factor)

    def test_stop_action_zero(self):
        self.fsm.update(is_ood=False, anomaly_score=0.0, threshold=0.05,
                        lidar_obs=self._lidar_close())
        v, omega = self.fsm.modify_action(1.0, 1.0)
        assert v == 0.0
        assert omega == 0.0

    def test_reset_returns_to_normal(self):
        self.fsm.update(is_ood=True, anomaly_score=0.1, threshold=0.05,
                        lidar_obs=self._lidar_clear())
        self.fsm.reset()
        assert self.fsm.state == RecoveryState.NORMAL


class TestSafetyController:
    def setup_method(self):
        self.sc = SafetyController(CFG, threshold=0.05, lidar_max_range=10.0)

    def _lidar_clear(self) -> np.ndarray:
        return np.ones(36, dtype=np.float32)

    def test_normal_passes_through(self):
        decision = self.sc.step(0.8, 0.3, self._lidar_clear(), anomaly_score=0.01)
        assert decision.state == RecoveryState.NORMAL
        assert not decision.is_ood
        assert not decision.ppo_overridden
        assert decision.v_cmd == pytest.approx(0.8)

    def test_ood_triggers_override(self):
        for _ in range(CFG.ood_confirmation_steps - 1):
            decision = self.sc.step(0.8, 0.3, self._lidar_clear(), anomaly_score=0.1)
            assert decision.state == RecoveryState.NORMAL

        decision = self.sc.step(0.8, 0.3, self._lidar_clear(), anomaly_score=0.1)
        assert decision.is_ood
        assert decision.ppo_overridden

    def test_isolated_ood_score_does_not_override(self):
        decision = self.sc.step(0.8, 0.3, self._lidar_clear(), anomaly_score=0.1)
        assert not decision.is_ood
        assert decision.state == RecoveryState.NORMAL
        assert not decision.ppo_overridden

    def test_rear_obstacle_does_not_trigger_safety_stop(self):
        config = SafetyConfig(
            critical_distance=0.5,
            stop_requires_ood=True,
            ood_confirmation_steps=1,
        )
        controller = SafetyController(
            config,
            threshold=0.05,
            lidar_max_range=10.0,
            forward_sector_fraction=0.5,
        )
        lidar = self._lidar_clear()
        lidar[0] = 0.01

        decision = controller.step(0.8, 0.3, lidar, anomaly_score=0.1)

        assert decision.state == RecoveryState.SLOWDOWN
        assert decision.min_obstacle_distance == pytest.approx(10.0)

    def test_ood_confirmation_resets_after_normal_score(self):
        self.sc.step(0.8, 0.3, self._lidar_clear(), anomaly_score=0.1)
        decision = self.sc.step(0.8, 0.3, self._lidar_clear(), anomaly_score=0.01)
        assert decision.state == RecoveryState.NORMAL

        for _ in range(CFG.ood_confirmation_steps - 1):
            decision = self.sc.step(0.8, 0.3, self._lidar_clear(), anomaly_score=0.1)
            assert decision.state == RecoveryState.NORMAL
        decision = self.sc.step(0.8, 0.3, self._lidar_clear(), anomaly_score=0.1)
        assert decision.is_ood

    def test_reset_clears_counters(self):
        self.sc.step(0.8, 0.3, self._lidar_clear(), anomaly_score=0.1)
        self.sc.reset()
        assert self.sc.step_count == 0
        assert self.sc.ood_detected_count == 0
