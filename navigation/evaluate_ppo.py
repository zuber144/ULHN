"""
navigation/evaluate_ppo.py
==========================
Evaluation loop for a trained PPO agent.

Measures:
    - Success rate
    - Collision rate
    - Average episode reward
    - Average path length
    - Average steps to goal
    - Average minimum obstacle distance
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from environment.ulhn_env import ULHNEnv

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Holds per-episode evaluation metrics."""
    episode: int
    seed: int
    success: bool
    collision: bool
    total_reward: float
    steps: int
    path_length: float
    min_obstacle_distance: float
    goal_distance_final: float


@dataclass
class EvalSummary:
    """Aggregated evaluation results across many episodes."""
    num_episodes: int
    success_rate: float
    collision_rate: float
    mean_reward: float
    std_reward: float
    mean_path_length: float
    mean_steps: float
    mean_min_obstacle_distance: float
    episodes: List[EpisodeResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("episodes")  # don't flatten all episode rows here
        return d

    def print_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY  (%d episodes)", self.num_episodes)
        logger.info("  Success rate          : %.1f%%", self.success_rate * 100)
        logger.info("  Collision rate        : %.1f%%", self.collision_rate * 100)
        logger.info("  Mean reward           : %.2f ± %.2f",
                    self.mean_reward, self.std_reward)
        logger.info("  Mean path length      : %.2f m", self.mean_path_length)
        logger.info("  Mean steps            : %.1f", self.mean_steps)
        logger.info("  Mean min obs dist     : %.2f m",
                    self.mean_min_obstacle_distance)
        logger.info("=" * 60)


def evaluate_ppo(
    model,
    config: dict,
    n_episodes: int = 50,
    base_seed: int = 1000,
    render: bool = False,
) -> EvalSummary:
    """
    Evaluate a trained PPO model for *n_episodes* episodes.

    Parameters
    ----------
    model:
        Loaded SB3 PPO model (or any object with .predict(obs) → (action, _)).
    config:
        Full config dict.
    n_episodes:
        Number of evaluation episodes.
    base_seed:
        Starting seed; each episode uses base_seed + episode_index.
    render:
        If True, render each step to screen.

    Returns
    -------
    EvalSummary
    """
    render_mode = "human" if render else None
    env = ULHNEnv(config=config, render_mode=render_mode)

    results: List[EpisodeResult] = []

    for ep in range(n_episodes):
        seed = base_seed + ep
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        path_length  = 0.0
        min_obs_dist = float("inf")
        prev_x, prev_y = info["x"], info["y"]
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Accumulate metrics
            total_reward += reward
            dx = info["x"] - prev_x
            dy = info["y"] - prev_y
            path_length += math.hypot(dx, dy)
            prev_x, prev_y = info["x"], info["y"]
            if info["min_obstacle_distance"] < min_obs_dist:
                min_obs_dist = info["min_obstacle_distance"]

            if render:
                env.render()

        result = EpisodeResult(
            episode=ep,
            seed=seed,
            success=info.get("goal_reached", False),
            collision=info.get("collision", False),
            total_reward=total_reward,
            steps=env.step_count,
            path_length=path_length,
            min_obstacle_distance=min_obs_dist if min_obs_dist < float("inf") else 0.0,
            goal_distance_final=info.get("goal_distance", 0.0),
        )
        results.append(result)

        logger.debug(
            "Ep %3d | seed=%d | success=%s | collision=%s | reward=%.2f | steps=%d",
            ep, seed, result.success, result.collision,
            result.total_reward, result.steps,
        )

    env.close()

    rewards       = [r.total_reward for r in results]
    path_lengths  = [r.path_length for r in results]
    steps_list    = [r.steps for r in results]
    min_dists     = [r.min_obstacle_distance for r in results]

    summary = EvalSummary(
        num_episodes=n_episodes,
        success_rate=sum(r.success for r in results) / n_episodes,
        collision_rate=sum(r.collision for r in results) / n_episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_path_length=float(np.mean(path_lengths)),
        mean_steps=float(np.mean(steps_list)),
        mean_min_obstacle_distance=float(np.mean(min_dists)),
        episodes=results,
    )

    summary.print_summary()
    return summary


def save_eval_results(
    summary: EvalSummary,
    label: str,
    results_dir: str = "results/metrics",
) -> None:
    """Save evaluation results to CSV and JSON."""
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Per-episode CSV
    rows = [asdict(r) for r in summary.episodes]
    df = pd.DataFrame(rows)
    csv_path = out / f"eval_{label}_episodes.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Episode results saved to %s", csv_path)

    # Summary JSON
    import json
    json_path = out / f"eval_{label}_summary.json"
    with json_path.open("w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    logger.info("Summary saved to %s", json_path)
