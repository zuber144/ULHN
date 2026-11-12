"""
navigation/callbacks.py
=======================
Custom Stable-Baselines3 callbacks for the ULHN PPO training loop.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

logger = logging.getLogger(__name__)


class ProgressCallback(BaseCallback):
    """
    Logs mean episode reward, length, success/collision rate to console.

    Works with SB3's ep_info_buffer (populated by VecEnv Monitor wrapper)
    AND by tracking raw terminal info dicts from ULHNEnv directly.
    """

    def __init__(self, log_interval_steps: int = 10_000, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_interval_steps = log_interval_steps
        self._last_log_step = 0
        self._successes: int = 0
        self._collisions: int = 0
        self._episodes: int = 0
        self._ep_rewards: list = []
        self._ep_lengths: list = []
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Primary: read from SB3's ep_info_buffer (populated when episode ends)
        if hasattr(self.model, "ep_info_buffer"):
            buf = self.model.ep_info_buffer
            if buf:
                # Count newly completed episodes since last step
                # ep_info_buffer is a deque; we track total via _episodes
                while len(self._ep_rewards) < len(buf):
                    entry = buf[len(self._ep_rewards)]
                    self._ep_rewards.append(entry.get("r", 0.0))
                    self._ep_lengths.append(entry.get("l", 0))
                    self._episodes += 1

        # Secondary: track success/collision from info dicts
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                if info.get("goal_reached", False):
                    self._successes += 1
                if info.get("collision", False):
                    self._collisions += 1

        if self.num_timesteps - self._last_log_step >= self.log_interval_steps:
            self._log()
            self._last_log_step = self.num_timesteps

        return True

    def _log(self) -> None:
        elapsed = time.time() - self._start_time
        fps = self.num_timesteps / max(elapsed, 1e-6)

        # Use last 100 completed episodes
        recent_rewards = self._ep_rewards[-100:] if self._ep_rewards else [0.0]
        recent_lengths = self._ep_lengths[-100:] if self._ep_lengths else [0]
        mean_reward = float(np.mean(recent_rewards))
        mean_len    = float(np.mean(recent_lengths))

        ep_count = self._episodes
        success_rate   = self._successes  / max(ep_count, 1)
        collision_rate = self._collisions / max(ep_count, 1)

        logger.info(
            "Step %7d | FPS %5.0f | ep_reward %7.2f | ep_len %5.1f | "
            "success %.1f%% | collision %.1f%% | episodes %d",
            self.num_timesteps, fps, mean_reward, mean_len,
            success_rate * 100, collision_rate * 100, ep_count,
        )

    def _on_training_end(self) -> None:
        self._log()


def make_checkpoint_callback(
    save_dir: str,
    save_freq: int = 50_000,
    name_prefix: str = "ulhn_ppo",
) -> CheckpointCallback:
    """Create a SB3 CheckpointCallback that saves to *save_dir*."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    return CheckpointCallback(
        save_freq=save_freq,
        save_path=str(save_dir),
        name_prefix=name_prefix,
        verbose=0,
    )
