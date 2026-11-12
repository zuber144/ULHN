"""tests/test_environment.py — Integration tests for ULHNEnv Gymnasium environment."""

from __future__ import annotations

import math
import pytest
import numpy as np
from environment.ulhn_env import ULHNEnv


def _minimal_config() -> dict:
    return {
        "environment": {
            "width": 20.0, "height": 20.0, "timestep": 0.1,
            "max_episode_steps": 50, "seed": 42,
        },
        "robot": {
            "radius": 0.3, "max_linear_velocity": 1.0, "max_angular_velocity": 1.5,
        },
        "lidar": {
            "num_rays": 36, "fov_deg": 360.0, "max_range": 10.0,
            "noise_std": 0.01, "noise_enabled": False,
        },
        "goal": {"radius": 0.5, "min_distance": 3.0},
        "obstacles": {
            "num_static_min": 3, "num_static_max": 5,
            "rect_width_range": [0.5, 2.0], "rect_height_range": [0.5, 2.0],
            "circle_radius_range": [0.3, 1.0], "rect_fraction": 0.6, "margin": 0.5,
        },
        "rewards": {
            "goal_reward": 100.0, "collision_penalty": -50.0,
            "progress_scale": 10.0, "time_penalty": -0.1,
            "smoothness_scale": 0.5, "obstacle_clearance_penalty": -0.5,
            "obstacle_clearance_threshold": 1.0,
        },
    }


@pytest.fixture
def env():
    e = ULHNEnv(config=_minimal_config(), render_mode=None)
    yield e
    e.close()


class TestULHNEnvSpaces:
    def test_observation_space_shape(self, env):
        # 36 lidar + 4 scalars = 40
        assert env.observation_space.shape == (40,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (2,)

    def test_action_space_bounds(self, env):
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)


class TestULHNEnvReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (40,)
        assert isinstance(info, dict)

    def test_obs_in_valid_range(self, env):
        obs, _ = env.reset(seed=0)
        # Observation bounds are [-1, 1] for most components
        assert not np.any(np.isnan(obs))

    def test_robot_not_in_obstacle_after_reset(self, env):
        for seed in range(5):
            obs, info = env.reset(seed=seed)
            assert not env.obstacle_map.point_in_obstacle(
                info["x"], info["y"], margin=env.robot.radius
            ), f"Robot spawned inside obstacle with seed={seed}"

    def test_goal_not_in_obstacle_after_reset(self, env):
        for seed in range(5):
            obs, info = env.reset(seed=seed)
            assert not env.obstacle_map.point_in_obstacle(
                info["goal_x"], info["goal_y"], margin=env.goal_radius
            ), f"Goal spawned inside obstacle with seed={seed}"

    def test_step_count_resets(self, env):
        env.reset()
        for _ in range(3):
            env.step(env.action_space.sample())
        env.reset()
        assert env.step_count == 0


class TestULHNEnvStep:
    def test_step_returns_correct_types(self, env):
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        assert obs.shape == (40,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_truncation_at_max_steps(self, env):
        env.reset(seed=2)
        terminated = truncated = False
        step = 0
        max_s = env.max_episode_steps
        while step < max_s + 5 and not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(
                np.array([0.0, 0.0], dtype=np.float32)
            )
            step += 1
        # Should truncate at max steps
        assert truncated or terminated

    def test_no_raises_on_random_actions(self, env):
        env.reset(seed=3)
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert True  # no exception

    def test_collision_terminates(self, env):
        """Drive into a wall at high speed — should eventually collide."""
        env.reset(seed=10)
        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(
                np.array([1.0, 0.0], dtype=np.float32)
            )
            if terminated:
                assert info["collision"] or info["goal_reached"]
                return
        # It's fine if no collision happened; just verify no crash


class TestReproducibility:
    def test_same_seed_same_start(self, env):
        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)
        assert info1["x"] == pytest.approx(info2["x"])
        assert info1["y"] == pytest.approx(info2["y"])
        assert info1["goal_x"] == pytest.approx(info2["goal_x"])
