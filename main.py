"""
main.py
=======
ULHN — Uncertainty-aware Learning Navigation
Complete command-line interface.

Usage:
    python main.py simulate
    python main.py train-ppo
    python main.py evaluate-ppo
    python main.py collect-lidar
    python main.py train-watchdog
    python main.py test-watchdog
    python main.py run-ulhn
    python main.py experiment --scenario lidar_noise
    python main.py compare
    streamlit run visualization/dashboard.py
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_ppo(config: dict) -> object:
    from navigation.train_ppo import load_ppo
    train_cfg = config.get("training", {})
    model_dir = Path(train_cfg.get("model_dir", "models/ppo"))
    path = model_dir / "ppo_final.zip"
    if not path.exists():
        raise FileNotFoundError(
            f"PPO model not found at {path}.\n"
            "Train first with:  python main.py train-ppo"
        )
    return load_ppo(str(path), config)


def _load_watchdog(config: dict):
    from watchdog.autoencoder import load_autoencoder
    from watchdog.threshold import load_threshold_stats
    from watchdog.anomaly_detector import AnomalyDetector

    wd_cfg = config.get("watchdog", {})
    model_dir = Path(wd_cfg.get("model_dir", "models/autoencoder"))
    ae_path = model_dir / "autoencoder.pt"
    th_path = model_dir / "threshold_stats.json"

    if not ae_path.exists() or not th_path.exists():
        raise FileNotFoundError(
            f"Watchdog model/stats not found in {model_dir}.\n"
            "Train first with:  python main.py train-watchdog"
        )

    model = load_autoencoder(str(ae_path))
    stats = load_threshold_stats(str(th_path))
    percentile = float(wd_cfg.get("threshold_percentile", 95.0))
    forward_fraction = float(wd_cfg.get("forward_sector_fraction", 0.5))
    smoothing_window = int(wd_cfg.get("smoothing_window", 3))
    return AnomalyDetector.from_stats(
        model,
        stats,
        percentile=percentile,
        smoothing_window=smoothing_window,
        forward_sector_fraction=forward_fraction,
    )


def _load_safety(config: dict, threshold: float):
    from safety.safety_controller import SafetyController
    from safety.safety_rules import load_safety_config
    lidar_max = config.get("lidar", {}).get("max_range", 10.0)
    forward_fraction = config.get("watchdog", {}).get(
        "forward_sector_fraction", 0.5
    )
    scfg = load_safety_config(config.get("safety", {}))
    return SafetyController(
        scfg,
        threshold=threshold,
        lidar_max_range=lidar_max,
        forward_sector_fraction=forward_fraction,
    )


# ---------------------------------------------------------------------------
# Phase 1: Simulation demo
# ---------------------------------------------------------------------------

def cmd_simulate(args: argparse.Namespace, config: dict) -> None:
    """Open-loop simulation demo."""
    from environment.ulhn_env import ULHNEnv

    logger = logging.getLogger("simulate")
    logger.info("Starting simulation demo …")

    env = ULHNEnv(config=config, render_mode="human")
    obs, info = env.reset(seed=config.get("environment", {}).get("seed", 42))

    logger.info(
        "World: %.0f×%.0f m | LiDAR rays: %d | Obs dim: %d",
        env.world_width, env.world_height,
        env.lidar.num_rays, env.observation_space.shape[0],
    )

    total_reward = 0.0
    done = False
    t = 0

    while not done and t < args.steps:
        v     = 0.6 + 0.3 * math.sin(t * 0.05)
        omega = 0.4 * math.sin(t * 0.12)
        action = np.array(
            [v / env.robot.max_linear_velocity,
             omega / env.robot.max_angular_velocity],
            dtype=np.float32,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        t += 1
        env.render()

        if t % 50 == 0:
            logger.info(
                "Step %3d | pos=(%.2f,%.2f) | goal_dist=%.2f | reward=%.2f",
                t, info["x"], info["y"], info["goal_distance"], reward,
            )
        if info["collision"]:
            logger.warning("Collision at step %d!", t)
            break
        if info["goal_reached"]:
            logger.info("Goal reached at step %d!", t)
            break

    logger.info("Episode finished | steps=%d | reward=%.2f", t, total_reward)
    input("\nPress ENTER to close …")
    env.close()


# ---------------------------------------------------------------------------
# Phase 4: Train PPO
# ---------------------------------------------------------------------------

def cmd_train_ppo(args: argparse.Namespace, config: dict) -> None:
    from navigation.train_ppo import train_ppo
    logger = logging.getLogger("train_ppo")
    logger.info("Starting PPO training …")
    train_ppo(config)


# ---------------------------------------------------------------------------
# Phase 5: Evaluate PPO
# ---------------------------------------------------------------------------

def cmd_evaluate_ppo(args: argparse.Namespace, config: dict) -> None:
    from navigation.evaluate_ppo import evaluate_ppo, save_eval_results
    logger = logging.getLogger("evaluate_ppo")

    model = _load_ppo(config)
    exp_cfg = config.get("experiment", {})
    n_ep = args.episodes or exp_cfg.get("num_eval_episodes", 50)

    logger.info("Evaluating PPO for %d episodes …", n_ep)
    summary = evaluate_ppo(model, config, n_episodes=n_ep, base_seed=1000)
    save_eval_results(summary, label="ppo_baseline",
                      results_dir=exp_cfg.get("results_dir", "results") + "/metrics")


# ---------------------------------------------------------------------------
# Phase 6: Collect normal LiDAR dataset
# ---------------------------------------------------------------------------

def cmd_collect_lidar(args: argparse.Namespace, config: dict) -> None:
    from watchdog.dataset import collect_normal_lidar, save_dataset
    logger = logging.getLogger("collect_lidar")

    model = _load_ppo(config)
    wd_cfg = config.get("watchdog", {})
    n_ep = args.episodes or 200

    logger.info("Collecting normal LiDAR dataset (%d episodes) …", n_ep)
    data = collect_normal_lidar(model, config, n_episodes=n_ep, base_seed=5000)

    dataset_dir = wd_cfg.get("dataset_dir", "data/normal_lidar")
    save_dataset(data, dataset_dir, metadata={
        "n_episodes": n_ep,
        "n_samples": len(data),
        "lidar_rays": data.shape[1] if data.ndim > 1 else 1,
    })
    logger.info("Dataset collection complete: %d samples", len(data))


# ---------------------------------------------------------------------------
# Phase 7: Train Autoencoder watchdog
# ---------------------------------------------------------------------------

def cmd_train_watchdog(args: argparse.Namespace, config: dict) -> None:
    from watchdog.train_watchdog import train_watchdog_pipeline
    logger = logging.getLogger("train_watchdog")
    logger.info("Training Autoencoder watchdog …")
    train_watchdog_pipeline(config)


# ---------------------------------------------------------------------------
# Phase 8: Test watchdog
# ---------------------------------------------------------------------------

def cmd_test_watchdog(args: argparse.Namespace, config: dict) -> None:
    from watchdog.dataset import load_dataset
    from experiments.disturbances import gaussian_noise, corrupt_sector
    import json

    logger = logging.getLogger("test_watchdog")
    watchdog = _load_watchdog(config)

    wd_cfg = config.get("watchdog", {})
    data = load_dataset(wd_cfg.get("dataset_dir", "data/normal_lidar"))

    logger.info("Testing watchdog on %d normal samples …", len(data))
    normal_scores = np.array([watchdog.score(s) for s in data])
    fp_rate = np.mean(normal_scores > watchdog.threshold)
    logger.info(
        "Normal data: mean_score=%.6f  threshold=%.6f  FP_rate=%.1f%%",
        normal_scores.mean(), watchdog.threshold, fp_rate * 100,
    )

    # Test on disturbed observations
    disturbed = np.array([gaussian_noise(s, std=0.2) for s in data[:500]])
    dist_scores = np.array([watchdog.score(s) for s in disturbed])
    tp_rate = np.mean(dist_scores > watchdog.threshold)
    logger.info(
        "Disturbed data (gaussian noise std=0.2): mean_score=%.6f  TP_rate=%.1f%%",
        dist_scores.mean(), tp_rate * 100,
    )

    results = {
        "threshold": watchdog.threshold,
        "normal_mean_score": float(normal_scores.mean()),
        "normal_fp_rate": float(fp_rate),
        "disturbed_mean_score": float(dist_scores.mean()),
        "disturbed_tp_rate": float(tp_rate),
    }
    out_path = Path("results/metrics/watchdog_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Watchdog test results saved to %s", out_path)


# ---------------------------------------------------------------------------
# Phase 11: Run full ULHN
# ---------------------------------------------------------------------------

def cmd_run_ulhn(args: argparse.Namespace, config: dict) -> None:
    from environment.ulhn_env import ULHNEnv
    from experiments.disturbances import gaussian_noise

    logger = logging.getLogger("run_ulhn")
    logger.info("Running full ULHN system …")

    ppo      = _load_ppo(config)
    watchdog = _load_watchdog(config)
    safety   = _load_safety(config, watchdog.threshold)

    env = ULHNEnv(config=config, render_mode="human")
    obs, info = env.reset(seed=config.get("environment", {}).get("seed", 42))
    watchdog.reset()
    safety.reset()

    exp_cfg = config.get("experiment", {})
    dist_start = exp_cfg.get("demo_disturbance_start_step", 100)
    num_rays = env.lidar.num_rays
    total_reward = 0.0
    t = 0
    done = False

    logger.info("Demo: disturbance injection at step %d", dist_start)

    while not done and t < config.get("environment", {}).get("max_episode_steps", 500):
        lidar_obs = obs[:num_rays].copy()
        dist_active = t >= dist_start
        if dist_active:
            lidar_obs = gaussian_noise(lidar_obs, std=0.2)
            obs_for_agent = obs.copy()
            obs_for_agent[:num_rays] = lidar_obs
        else:
            obs_for_agent = obs

        action, _ = ppo.predict(obs_for_agent, deterministic=True)
        ppo_v     = float(action[0]) * env.robot.max_linear_velocity
        ppo_omega = float(action[1]) * env.robot.max_angular_velocity

        anomaly_score = watchdog.update(lidar_obs)
        decision = safety.step(ppo_v, ppo_omega, lidar_obs, anomaly_score)

        act = np.array([
            np.clip(decision.v_cmd / env.robot.max_linear_velocity, -1, 1),
            np.clip(decision.omega_cmd / env.robot.max_angular_velocity, -1, 1),
        ], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(act)
        total_reward += reward
        done = terminated or truncated
        t += 1
        env.render()

        if t % 20 == 0 or decision.is_ood:
            logger.info(
                "Step %3d | state=%-8s | anomaly=%.4f | thresh=%.4f | "
                "ood=%s | dist=%s",
                t, decision.state.name, anomaly_score, watchdog.threshold,
                decision.is_ood, dist_active,
            )

        if info["collision"]:
            logger.warning("Collision at step %d!", t)
            break
        if info["goal_reached"]:
            logger.info("Goal reached at step %d!", t)
            break

    logger.info(
        "ULHN demo finished | steps=%d | reward=%.2f | "
        "ood_detections=%d | emergency_stops=%d",
        t, total_reward,
        safety.ood_detected_count, safety.emergency_stop_count,
    )
    input("\nPress ENTER to close …")
    env.close()


# ---------------------------------------------------------------------------
# Phase 12/13: Run all experiments and compare
# ---------------------------------------------------------------------------

def cmd_experiment(args: argparse.Namespace, config: dict) -> None:
    from experiments.runner import run_experiment
    from evaluation.metrics import compute_metrics, save_metrics, print_comparison_table

    logger = logging.getLogger("experiment")
    scenario = args.scenario
    logger.info("Running experiment: scenario=%s", scenario)

    ppo = _load_ppo(config)

    try:
        watchdog = _load_watchdog(config)
        safety   = _load_safety(config, watchdog.threshold)
    except FileNotFoundError as e:
        logger.warning("Watchdog/safety not available: %s", e)
        watchdog = safety = None

    exp_cfg = config.get("experiment", {})
    n_ep    = args.episodes or exp_cfg.get("num_eval_episodes", 50)
    dist_start = exp_cfg.get("demo_disturbance_start_step", 100)
    results_dir = str(Path(exp_cfg.get("results_dir", "results")) / "metrics")

    all_metrics = []

    results_A = run_experiment(
        "A", scenario, config, ppo,
        anomaly_detector=None, safety_controller=None,
        n_episodes=n_ep, disturbance_start_step=dist_start,
    )
    all_metrics.append(compute_metrics(results_A, "A", scenario))

    if watchdog:
        watchdog.reset()
        results_B = run_experiment(
            "B", scenario, config, ppo,
            anomaly_detector=watchdog, safety_controller=None,
            n_episodes=n_ep, disturbance_start_step=dist_start,
        )
        all_metrics.append(compute_metrics(results_B, "B", scenario))

        watchdog.reset()
        safety.reset()
        results_C = run_experiment(
            "C", scenario, config, ppo,
            anomaly_detector=watchdog, safety_controller=safety,
            n_episodes=n_ep, disturbance_start_step=dist_start,
        )
        all_metrics.append(compute_metrics(results_C, "C", scenario))

    save_metrics(all_metrics, results_dir)
    print_comparison_table(all_metrics)


def cmd_compare(args: argparse.Namespace, config: dict) -> None:
    from experiments.runner import run_experiment
    from evaluation.metrics import compute_metrics, save_metrics, print_comparison_table
    from evaluation.plots import (
        plot_success_rate, plot_collision_rate, plot_comparison_summary
    )

    logger = logging.getLogger("compare")
    logger.info("Running full comparison across all scenarios …")

    ppo = _load_ppo(config)
    try:
        watchdog = _load_watchdog(config)
        safety   = _load_safety(config, watchdog.threshold)
    except FileNotFoundError as e:
        logger.error("Cannot compare without trained watchdog: %s", e)
        return

    scenarios = ["normal", "gaussian_noise", "missing_readings",
                 "corrupt_sector", "partial_blockage"]
    exp_cfg    = config.get("experiment", {})
    n_ep       = exp_cfg.get("num_eval_episodes", 20)
    dist_start = exp_cfg.get("demo_disturbance_start_step", 100)
    results_dir = str(Path(exp_cfg.get("results_dir", "results")) / "metrics")

    all_metrics = []

    for sc in scenarios:
        logger.info("Running scenario: %s", sc)
        for label, wd, sf in [("A", None, None), ("B", watchdog, None), ("C", watchdog, safety)]:
            if wd:
                wd.reset()
            if sf:
                sf.reset()
            res = run_experiment(
                label, sc, config, ppo,
                anomaly_detector=wd, safety_controller=sf,
                n_episodes=n_ep, disturbance_start_step=dist_start,
            )
            all_metrics.append(compute_metrics(res, label, sc))

    save_metrics(all_metrics, results_dir)
    print_comparison_table(all_metrics)

    fig_dir = str(Path(exp_cfg.get("results_dir", "results")) / "figures")
    plot_success_rate(all_metrics, f"{fig_dir}/success_rate.png")
    plot_collision_rate(all_metrics, f"{fig_dir}/collision_rate.png")
    plot_comparison_summary(all_metrics, f"{fig_dir}/comparison_summary.png")
    logger.info("All plots saved to %s", fig_dir)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ulhn",
        description="ULHN — Uncertainty-aware Learning Navigation",
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = p.add_subparsers(dest="command")

    sim = sub.add_parser("simulate", help="Phase 1: Run 2D simulator demo")
    sim.add_argument("--steps", type=int, default=300)

    train = sub.add_parser("train-ppo", help="Phase 4: Train PPO agent")

    ev = sub.add_parser("evaluate-ppo", help="Phase 5: Evaluate trained PPO")
    ev.add_argument("--episodes", type=int, default=None)

    cl = sub.add_parser("collect-lidar", help="Phase 6: Collect normal LiDAR dataset")
    cl.add_argument("--episodes", type=int, default=None)

    sub.add_parser("train-watchdog", help="Phase 7: Train Autoencoder watchdog")
    sub.add_parser("test-watchdog",  help="Phase 8: Test anomaly detection")
    sub.add_parser("run-ulhn",       help="Phase 11: Run full ULHN system")
    sub.add_parser("compare",        help="Phase 13: Full system comparison")

    ex = sub.add_parser("experiment", help="Run named experiment")
    ex.add_argument("--scenario", required=True,
                    choices=["normal", "gaussian_noise", "missing_readings",
                             "corrupt_sector", "partial_blockage"])
    ex.add_argument("--episodes", type=int, default=None)

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(getattr(args, "log_level", "INFO"))
    config = _load_config(getattr(args, "config", "config.yaml"))

    dispatch = {
        "simulate":      cmd_simulate,
        "train-ppo":     cmd_train_ppo,
        "evaluate-ppo":  cmd_evaluate_ppo,
        "collect-lidar": cmd_collect_lidar,
        "train-watchdog":cmd_train_watchdog,
        "test-watchdog": cmd_test_watchdog,
        "run-ulhn":      cmd_run_ulhn,
        "compare":       cmd_compare,
        "experiment":    cmd_experiment,
    }

    if args.command in dispatch:
        dispatch[args.command](args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
