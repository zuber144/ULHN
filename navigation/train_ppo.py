"""
navigation/train_ppo.py
=======================
PPO training script for ULHN navigation.

Usage (via CLI):
    python main.py train-ppo

Or directly:
    python -m navigation.train_ppo --config config.yaml
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList


from environment.ulhn_env import ULHNEnv
from navigation.policy import get_policy_kwargs
from navigation.callbacks import ProgressCallback, make_checkpoint_callback

logger = logging.getLogger(__name__)


def set_global_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env_fn(config: dict, rank: int, base_seed: int, log_dir: Optional[Path] = None):
    """Factory that returns a thunk creating a seeded, monitored ULHNEnv."""
    def _init():
        env = ULHNEnv(config=config)
        env.reset(seed=base_seed + rank)
        # Wrap with Monitor so SB3 fills ep_info_buffer for reward logging
        monitor_path = None
        if log_dir is not None:
            monitor_path = str(log_dir / f"monitor_{rank}")
        env = Monitor(env, filename=monitor_path)
        return env
    return _init


def train_ppo(config: dict) -> PPO:
    """
    Train a PPO agent on the ULHN environment.

    Parameters
    ----------
    config:
        Full config dict from config.yaml.

    Returns
    -------
    Trained PPO model.
    """
    train_cfg = config.get("training", {})
    seed       = train_cfg.get("seed", 42)
    total_ts   = train_cfg.get("total_timesteps", 500_000)
    n_envs     = train_cfg.get("n_envs", 4)
    lr         = train_cfg.get("learning_rate", 3e-4)
    batch_size = train_cfg.get("batch_size", 64)
    n_epochs   = train_cfg.get("n_epochs", 10)
    gamma      = train_cfg.get("gamma", 0.99)
    gae_lambda = train_cfg.get("gae_lambda", 0.95)
    clip_range = train_cfg.get("clip_range", 0.2)
    ent_coef   = train_cfg.get("ent_coef", 0.01)
    log_dir    = Path(train_cfg.get("log_dir", "logs/ppo"))
    model_dir  = Path(train_cfg.get("model_dir", "models/ppo"))

    net_arch    = train_cfg.get("net_arch", [256, 256])
    activation  = train_cfg.get("activation_fn", "tanh")

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    set_global_seeds(seed)

    logger.info("Creating %d parallel environments (seed base=%d)", n_envs, seed)

    # Use DummyVecEnv (works on Windows — SubprocVecEnv needs multiprocessing guards)
    env_fns = [make_env_fn(config, i, seed, log_dir) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)   # aggregates episode stats into ep_info_buffer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training device: %s", device)

    policy_kwargs = get_policy_kwargs(net_arch=net_arch, activation_fn=activation)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=lr,
        n_steps=max(batch_size * 4, 2048) // n_envs,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_dir),
        verbose=0,
        seed=seed,
        device=device,
    )

    callbacks = CallbackList([
        ProgressCallback(log_interval_steps=10_000),
        make_checkpoint_callback(
            save_dir=str(model_dir / "checkpoints"),
            save_freq=max(50_000 // n_envs, 1),
        ),
    ])

    logger.info("Starting PPO training for %d timesteps …", total_ts)
    model.learn(
        total_timesteps=total_ts,
        callback=callbacks,
        tb_log_name="ppo_ulhn",
        reset_num_timesteps=True,
    )

    # Save final model
    final_path = model_dir / "ppo_final"
    model.save(str(final_path))
    logger.info("Training complete. Model saved to %s.zip", final_path)

    vec_env.close()
    return model


def load_ppo(model_path: str, config: dict) -> PPO:
    """
    Load a previously trained PPO model.

    Parameters
    ----------
    model_path:
        Path to the .zip file (with or without extension).
    config:
        Full config dict.

    Returns
    -------
    Loaded PPO model.
    """
    path = Path(model_path)
    if not path.suffix:
        path = path.with_suffix(".zip")
    if not path.exists():
        raise FileNotFoundError(f"PPO model not found: {path.resolve()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(str(path), device=device)
    logger.info("Loaded PPO model from %s", path)
    return model
