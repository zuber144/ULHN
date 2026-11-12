"""
experiments/runner.py
=====================
Experiment runner that evaluates a system (PPO, PPO+Watchdog, ULHN)
across normal and disturbed scenarios.

Experiments A, B, C:
    A: PPO only
    B: PPO + Autoencoder Watchdog (detect but no recovery action)
    C: PPO + Autoencoder + Safety Recovery (full ULHN)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from environment.ulhn_env import ULHNEnv
from experiments.disturbances import (
    gaussian_noise,
    missing_readings,
    corrupt_sector,
    partial_blockage,
)

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """Per-timestep log entry."""
    episode: int
    step: int
    x: float
    y: float
    goal_x: float
    goal_y: float
    lidar_obs: List[float]
    ppo_v: float
    ppo_omega: float
    final_v: float
    final_omega: float
    anomaly_score: float
    threshold: float
    safety_state: str
    disturbance_active: bool
    collision: bool
    goal_reached: bool


@dataclass
class ExperimentResult:
    """Per-episode result for an experiment."""
    experiment: str      # 'A', 'B', or 'C'
    scenario: str        # 'normal', 'lidar_noise', etc.
    episode: int
    seed: int
    success: bool
    collision: bool
    total_reward: float
    steps: int
    path_length: float
    min_obstacle_distance: float
    ood_detected_count: int
    emergency_stop_count: int
    recovery_count: int
    max_anomaly_score: float
    disturbance_started_at: int


DISTURBANCE_FNS: Dict[str, Callable] = {
    "gaussian_noise":    gaussian_noise,
    "missing_readings":  missing_readings,
    "corrupt_sector":    lambda obs, **kw: corrupt_sector(obs, start_ray=0, width_rays=9),
    "partial_blockage":  lambda obs, **kw: partial_blockage(obs, centre_ray=0, half_width=6),
    "normal":            lambda obs, **kw: obs,
}


def run_experiment(
    experiment_label: str,     # 'A', 'B', or 'C'
    scenario: str,             # 'normal', 'gaussian_noise', etc.
    config: dict,
    ppo_model,
    anomaly_detector=None,     # None for experiment A
    safety_controller=None,    # None for experiments A, B
    n_episodes: int = 50,
    base_seed: int = 2000,
    disturbance_start_step: int = 100,
    save_step_logs: bool = False,
    results_dir: str = "results/experiment_logs",
) -> List[ExperimentResult]:
    """
    Run one experiment configuration for *n_episodes*.

    Parameters
    ----------
    experiment_label:
        'A', 'B', or 'C'.
    scenario:
        One of 'normal', 'gaussian_noise', 'missing_readings',
        'corrupt_sector', 'partial_blockage'.
    config:
        Full config dict.
    ppo_model:
        Loaded SB3 PPO model.
    anomaly_detector:
        AnomalyDetector instance (None for experiment A).
    safety_controller:
        SafetyController instance (None for A, B).
    n_episodes:
        Number of episodes to run.
    base_seed:
        Starting seed for reproducibility.
    disturbance_start_step:
        Step at which the disturbance is injected (0 = always active).
    save_step_logs:
        If True, save full per-step logs to JSON.
    results_dir:
        Directory to save logs.

    Returns
    -------
    List of ExperimentResult (one per episode).
    """
    lidar_cfg = config.get("lidar", {})
    num_rays  = lidar_cfg.get("num_rays", 36)
    max_range = lidar_cfg.get("max_range", 10.0)

    disturb_fn = DISTURBANCE_FNS.get(scenario, DISTURBANCE_FNS["normal"])
    rng = np.random.default_rng(base_seed)

    env = ULHNEnv(config=config, render_mode=None)
    results: List[ExperimentResult] = []

    threshold = getattr(anomaly_detector, "threshold", 0.0) if anomaly_detector else 0.0

    for ep in range(n_episodes):
        seed = base_seed + ep
        obs, info = env.reset(seed=seed)

        if anomaly_detector is not None:
            anomaly_detector.reset()
        if safety_controller is not None:
            safety_controller.reset()

        total_reward   = 0.0
        path_length    = 0.0
        min_obs_dist   = float("inf")
        ood_count      = 0
        stop_count     = 0
        recovery_count = 0
        max_score      = 0.0
        prev_x, prev_y = info["x"], info["y"]
        prev_safety    = "NORMAL"

        step_logs: List[StepRecord] = []
        done = False
        step = 0
        t0 = time.perf_counter()

        while not done:
            # Extract LiDAR from obs
            lidar_obs = obs[:num_rays].copy()
            disturbance_active = (step >= disturbance_start_step and
                                  scenario != "normal")

            # Apply disturbance to the observation seen by the watchdog/agent
            if disturbance_active:
                disturbed_lidar = disturb_fn(lidar_obs, rng=rng)
                # Rebuild full observation with disturbed LiDAR
                obs_for_agent = obs.copy()
                obs_for_agent[:num_rays] = disturbed_lidar
            else:
                disturbed_lidar = lidar_obs
                obs_for_agent = obs

            # PPO decision
            action, _ = ppo_model.predict(obs_for_agent, deterministic=True)
            ppo_v     = float(action[0]) * env.robot.max_linear_velocity
            ppo_omega = float(action[1]) * env.robot.max_angular_velocity

            # Watchdog
            anomaly_score = 0.0
            if anomaly_detector is not None:
                anomaly_score = anomaly_detector.update(disturbed_lidar)
                if anomaly_score > max_score:
                    max_score = anomaly_score
                if anomaly_detector.is_ood():
                    ood_count += 1

            # Safety controller
            safety_state_str = "NORMAL"
            final_v, final_omega = ppo_v, ppo_omega
            if safety_controller is not None:
                decision = safety_controller.step(
                    ppo_v=ppo_v,
                    ppo_omega=ppo_omega,
                    lidar_obs=disturbed_lidar,
                    anomaly_score=anomaly_score,
                )
                final_v      = decision.v_cmd
                final_omega  = decision.omega_cmd
                safety_state_str = decision.state.name

                if decision.state.name == "STOP":
                    stop_count += 1
                if prev_safety != "NORMAL" and decision.state.name == "NORMAL":
                    recovery_count += 1
                prev_safety = safety_state_str

            # Scale back to [-1,1] for env
            act_v = np.clip(final_v / env.robot.max_linear_velocity, -1, 1)
            act_w = np.clip(final_omega / env.robot.max_angular_velocity, -1, 1)
            action_final = np.array([act_v, act_w], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action_final)
            done = terminated or truncated
            total_reward += reward
            step += 1

            dx = info["x"] - prev_x
            dy = info["y"] - prev_y
            path_length += math.hypot(dx, dy)
            prev_x, prev_y = info["x"], info["y"]

            if info["min_obstacle_distance"] < min_obs_dist:
                min_obs_dist = info["min_obstacle_distance"]

            if save_step_logs:
                step_logs.append(StepRecord(
                    episode=ep, step=step,
                    x=info["x"], y=info["y"],
                    goal_x=info["goal_x"], goal_y=info["goal_y"],
                    lidar_obs=disturbed_lidar.tolist(),
                    ppo_v=ppo_v, ppo_omega=ppo_omega,
                    final_v=final_v, final_omega=final_omega,
                    anomaly_score=anomaly_score,
                    threshold=threshold,
                    safety_state=safety_state_str,
                    disturbance_active=disturbance_active,
                    collision=info.get("collision", False),
                    goal_reached=info.get("goal_reached", False),
                ))

        result = ExperimentResult(
            experiment=experiment_label,
            scenario=scenario,
            episode=ep,
            seed=seed,
            success=info.get("goal_reached", False),
            collision=info.get("collision", False),
            total_reward=total_reward,
            steps=step,
            path_length=path_length,
            min_obstacle_distance=min_obs_dist if min_obs_dist < float("inf") else 0.0,
            ood_detected_count=ood_count,
            emergency_stop_count=stop_count,
            recovery_count=recovery_count,
            max_anomaly_score=max_score,
            disturbance_started_at=disturbance_start_step,
        )
        results.append(result)

        if save_step_logs and step_logs:
            _save_step_log(step_logs, results_dir, experiment_label, scenario, ep)

    env.close()

    # Summary
    n = len(results)
    logger.info(
        "Exp %s | scenario=%-20s | success=%.0f%% | collision=%.0f%% | "
        "ood_per_ep=%.1f | stops_per_ep=%.1f",
        experiment_label,
        scenario,
        100 * sum(r.success for r in results) / n,
        100 * sum(r.collision for r in results) / n,
        sum(r.ood_detected_count for r in results) / n,
        sum(r.emergency_stop_count for r in results) / n,
    )

    return results


def _save_step_log(
    logs: List[StepRecord],
    results_dir: str,
    exp: str,
    scenario: str,
    episode: int,
) -> None:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"exp{exp}_{scenario}_ep{episode:03d}.jsonl"
    with path.open("w") as f:
        for rec in logs:
            f.write(json.dumps(asdict(rec)) + "\n")
