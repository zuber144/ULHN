"""
watchdog/dataset.py
===================
Normal LiDAR dataset collection and management.

Runs a trained PPO agent in a clean (undisturbed) environment and records
every LiDAR observation. These observations form the training distribution
for the Autoencoder watchdog.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from environment.ulhn_env import ULHNEnv

logger = logging.getLogger(__name__)


def collect_normal_lidar(
    model,
    config: dict,
    n_episodes: int = 200,
    base_seed: int = 5000,
    max_steps_per_episode: int = 500,
) -> np.ndarray:
    """
    Collect normal LiDAR observations by running the trained PPO agent.

    Only observations from episodes where the robot does NOT collide are
    included in the dataset to keep it "normal".

    Parameters
    ----------
    model:
        Trained SB3 PPO (or any object with .predict(obs)).
    config:
        Full config dict.
    n_episodes:
        Number of episodes to run.
    base_seed:
        Seed offset (different from training seeds).
    max_steps_per_episode:
        Steps hard cap per episode.

    Returns
    -------
    np.ndarray of shape (N, num_rays), dtype float32
        All collected normal LiDAR observations.
    """
    env = ULHNEnv(config=config, render_mode=None)
    num_rays = env.lidar.num_rays

    all_observations = []
    episode_observations = []

    for ep in range(n_episodes):
        seed = base_seed + ep
        obs, info = env.reset(seed=seed)
        episode_observations.clear()
        collision = False

        for _ in range(max_steps_per_episode):
            # Extract just the LiDAR part of the observation
            lidar_obs = obs[:num_rays].copy()
            episode_observations.append(lidar_obs)

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)

            if info.get("collision", False):
                collision = True
            if terminated or truncated:
                break

        # Only include episodes without collision
        if not collision:
            all_observations.extend(episode_observations)
            logger.debug(
                "Ep %3d: collected %d normal observations (seed=%d)",
                ep, len(episode_observations), seed,
            )
        else:
            logger.debug("Ep %3d: skipped (collision)", ep)

    env.close()

    dataset = np.array(all_observations, dtype=np.float32)
    logger.info(
        "Normal LiDAR dataset: %d samples from %d episodes",
        len(dataset), n_episodes,
    )
    return dataset


def save_dataset(
    data: np.ndarray,
    dataset_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save dataset as compressed NumPy archive with metadata."""
    out = Path(dataset_dir)
    out.mkdir(parents=True, exist_ok=True)

    npy_path = out / "normal_lidar.npy"
    np.save(str(npy_path), data)

    if metadata is not None:
        import json
        meta_path = out / "metadata.json"
        with meta_path.open("w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Metadata saved to %s", meta_path)

    logger.info("Dataset saved: %s  shape=%s", npy_path, data.shape)


def load_dataset(dataset_dir: str) -> np.ndarray:
    """Load a previously saved normal LiDAR dataset."""
    p = Path(dataset_dir) / "normal_lidar.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"LiDAR dataset not found at {p.resolve()}. "
            "Run: python main.py collect-lidar"
        )
    data = np.load(str(p)).astype(np.float32)
    logger.info("Dataset loaded from %s  shape=%s", p, data.shape)
    return data
