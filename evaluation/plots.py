"""
evaluation/plots.py
===================
All required publication-quality plots for the ULHN project.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

PALETTE = {
    "A":  "#4488ff",   # PPO only — blue
    "B":  "#ffaa00",   # PPO + Watchdog — amber
    "C":  "#44dd88",   # Full ULHN — green
    "threshold": "#ff4444",
    "normal":    "#44aaff",
    "disturbed": "#ff8844",
}


def _savefig(fig: plt.Figure, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


# ---------------------------------------------------------------------------
# Plot 1: Training reward curve
# ---------------------------------------------------------------------------

def plot_training_reward(
    log_dir: str,
    out_path: str = "results/figures/training_reward.png",
) -> None:
    """Parse TensorBoard CSV and plot reward vs timestep."""
    try:
        import pandas as pd
        csvs = list(Path(log_dir).rglob("*monitor*.csv"))
        if not csvs:
            logger.warning("No monitor CSV found in %s", log_dir)
            return
        dfs = [pd.read_csv(f, comment="#") for f in csvs]
        df = pd.concat(dfs, ignore_index=True).sort_values("t")
        window = min(50, len(df) // 5 + 1)
        smoothed = df["r"].rolling(window, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f0f1e")
        ax.set_facecolor("#1a1a2e")
        ax.plot(df["t"], df["r"], alpha=0.25, color=PALETTE["A"], lw=0.8)
        ax.plot(df["t"], smoothed, color=PALETTE["A"], lw=2, label="Smoothed reward")
        ax.set_xlabel("Timestep", color="#aaa")
        ax.set_ylabel("Episode Reward", color="#aaa")
        ax.set_title("PPO Training Reward vs Timestep", color="#ccf", fontsize=13)
        ax.legend(facecolor="#222244", labelcolor="#ccf")
        ax.tick_params(colors="#aaa")
        _savefig(fig, Path(out_path), "Training reward")
    except Exception as e:
        logger.warning("Could not plot training reward: %s", e)


# ---------------------------------------------------------------------------
# Plot 2: Success rate comparison
# ---------------------------------------------------------------------------

def plot_success_rate(
    metrics: list,
    out_path: str = "results/figures/success_rate.png",
) -> None:
    systems  = sorted(set(m.system for m in metrics))
    scenarios = sorted(set(m.scenario for m in metrics))

    x = np.arange(len(scenarios))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0f0f1e")
    ax.set_facecolor("#1a1a2e")

    for i, sys in enumerate(systems):
        vals = []
        for sc in scenarios:
            ms = [m for m in metrics if m.system == sys and m.scenario == sc]
            vals.append(ms[0].success_rate * 100 if ms else 0.0)
        bars = ax.bar(x + i * w, vals, w, label=f"System {sys}",
                      color=PALETTE.get(sys, "#888"), alpha=0.85)

    ax.set_xticks(x + w)
    ax.set_xticklabels(scenarios, rotation=20, ha="right", color="#aaa")
    ax.set_ylabel("Success Rate (%)", color="#aaa")
    ax.set_title("Success Rate: PPO vs PPO+Watchdog vs ULHN", color="#ccf", fontsize=13)
    ax.legend(facecolor="#222244", labelcolor="#ccf")
    ax.tick_params(colors="#aaa")
    ax.set_ylim(0, 110)
    _savefig(fig, Path(out_path), "Success rate")


# ---------------------------------------------------------------------------
# Plot 3: Collision rate
# ---------------------------------------------------------------------------

def plot_collision_rate(
    metrics: list,
    out_path: str = "results/figures/collision_rate.png",
) -> None:
    systems  = sorted(set(m.system for m in metrics))
    scenarios = sorted(set(m.scenario for m in metrics))
    x = np.arange(len(scenarios))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0f0f1e")
    ax.set_facecolor("#1a1a2e")
    for i, sys in enumerate(systems):
        vals = []
        for sc in scenarios:
            ms = [m for m in metrics if m.system == sys and m.scenario == sc]
            vals.append(ms[0].collision_rate * 100 if ms else 0.0)
        ax.bar(x + i * w, vals, w, label=f"System {sys}",
               color=PALETTE.get(sys, "#888"), alpha=0.85)

    ax.set_xticks(x + w)
    ax.set_xticklabels(scenarios, rotation=20, ha="right", color="#aaa")
    ax.set_ylabel("Collision Rate (%)", color="#aaa")
    ax.set_title("Collision Rate by System and Scenario", color="#ccf", fontsize=13)
    ax.legend(facecolor="#222244", labelcolor="#ccf")
    ax.tick_params(colors="#aaa")
    _savefig(fig, Path(out_path), "Collision rate")


# ---------------------------------------------------------------------------
# Plot 4: Path length comparison (box plots)
# ---------------------------------------------------------------------------

def plot_path_length(
    all_results: dict,   # {('A','normal'): [ExperimentResult], ...}
    out_path: str = "results/figures/path_length.png",
) -> None:
    labels = []
    data   = []
    colors = []

    for (sys, sc), results in sorted(all_results.items()):
        labels.append(f"{sys}\n{sc[:8]}")
        data.append([r.path_length for r in results])
        colors.append(PALETTE.get(sys, "#888"))

    fig, ax = plt.subplots(figsize=(max(8, len(data) * 1.2), 5), facecolor="#0f0f1e")
    ax.set_facecolor("#1a1a2e")
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, color="#aaa", fontsize=8)
    ax.set_ylabel("Path Length (m)", color="#aaa")
    ax.set_title("Path Length Distribution", color="#ccf", fontsize=13)
    ax.tick_params(colors="#aaa")
    _savefig(fig, Path(out_path), "Path length")


# ---------------------------------------------------------------------------
# Plot 5: Anomaly score over time (single episode)
# ---------------------------------------------------------------------------

def plot_anomaly_timeline(
    scores: List[float],
    threshold: float,
    disturbance_start: int = 100,
    out_path: str = "results/figures/anomaly_timeline.png",
) -> None:
    steps = list(range(len(scores)))
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0f0f1e")
    ax.set_facecolor("#1a1a2e")
    ax.plot(steps, scores, color=PALETTE["normal"], lw=1.5, label="Anomaly score")
    ax.axhline(threshold, color=PALETTE["threshold"], lw=2, ls="--", label="Threshold")
    if disturbance_start < len(scores):
        ax.axvline(disturbance_start, color="#ff8844", lw=1.5, ls=":", label="Disturbance start")
    ax.fill_between(steps, scores, threshold,
                    where=[s > threshold for s in scores],
                    color="#ff4444", alpha=0.2, label="OOD region")
    ax.set_xlabel("Step", color="#aaa")
    ax.set_ylabel("Anomaly Score (MSE)", color="#aaa")
    ax.set_title("Anomaly Score vs Threshold Over Time", color="#ccf", fontsize=13)
    ax.legend(facecolor="#222244", labelcolor="#ccf")
    ax.tick_params(colors="#aaa")
    _savefig(fig, Path(out_path), "Anomaly timeline")


# ---------------------------------------------------------------------------
# Plot 6: Trajectory visualisation
# ---------------------------------------------------------------------------

def plot_trajectory(
    trajectory: List,
    obstacles: list,
    goal_pos: tuple,
    world_size: tuple = (20, 20),
    collision_step: Optional[int] = None,
    recovery_steps: Optional[List[int]] = None,
    out_path: str = "results/figures/trajectory.png",
) -> None:
    from environment.obstacles import RectObstacle, CircleObstacle

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0f0f1e")
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(-0.5, world_size[0] + 0.5)
    ax.set_ylim(-0.5, world_size[1] + 0.5)
    ax.set_aspect("equal")

    # Obstacles
    for obs in obstacles:
        if isinstance(obs, RectObstacle):
            rect = mpatches.Rectangle(
                (obs.cx - obs.half_w, obs.cy - obs.half_h),
                2 * obs.half_w, 2 * obs.half_h,
                edgecolor="#ff6644", facecolor="#3d1a0e", lw=1.5, zorder=3,
            )
            ax.add_patch(rect)
        elif isinstance(obs, CircleObstacle):
            circ = plt.Circle((obs.cx, obs.cy), obs.radius,
                               edgecolor="#ff6644", facecolor="#3d1a0e", lw=1.5, zorder=3)
            ax.add_patch(circ)

    # Trajectory
    if trajectory:
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
        ax.plot(xs, ys, color="#4488ff", lw=2, alpha=0.8, zorder=5)
        ax.plot(xs[0], ys[0], "o", markersize=10, color="#ffaa00", zorder=7, label="Start")

    # Goal
    gx, gy = goal_pos
    ax.plot(gx, gy, "*", markersize=16, color="#00ff88", zorder=7, label="Goal")

    # Recovery steps
    if recovery_steps and trajectory:
        for s in recovery_steps:
            if s < len(trajectory):
                ax.plot(*trajectory[s], "^", markersize=9,
                        color="#ffaa00", alpha=0.8, zorder=6)

    # Collision step
    if collision_step is not None and trajectory and collision_step < len(trajectory):
        ax.plot(*trajectory[collision_step], "x", markersize=14,
                color="#ff0000", lw=3, zorder=8, label="Collision")

    ax.set_title("Robot Trajectory", color="#ccf", fontsize=13)
    ax.legend(facecolor="#222244", labelcolor="#ccf")
    ax.tick_params(colors="#aaa")
    _savefig(fig, Path(out_path), "Trajectory")


# ---------------------------------------------------------------------------
# Plot 7: Overall performance comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison_summary(
    metrics: list,
    out_path: str = "results/figures/comparison_summary.png",
) -> None:
    normal_metrics = [m for m in metrics if m.scenario == "normal"]
    if not normal_metrics:
        normal_metrics = metrics

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#0f0f1e")
    fig.patch.set_facecolor("#0f0f1e")

    kv_list = [
        ("success_rate", "Success Rate (%)", 100),
        ("collision_rate", "Collision Rate (%)", 100),
        ("mean_path_length", "Mean Path Length (m)", 1),
    ]

    for ax, (attr, ylabel, scale) in zip(axes, kv_list):
        ax.set_facecolor("#1a1a2e")
        systems = [m.system for m in normal_metrics]
        vals    = [getattr(m, attr) * scale for m in normal_metrics]
        colors  = [PALETTE.get(s, "#888") for s in systems]
        bars = ax.bar(systems, vals, color=colors, alpha=0.85)
        ax.set_ylabel(ylabel, color="#aaa")
        ax.set_title(ylabel, color="#ccf", fontsize=11)
        ax.tick_params(colors="#aaa")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom",
                    color="#fff", fontsize=9)

    fig.suptitle("PPO vs PPO+Watchdog vs ULHN — Normal Environment",
                 color="#ccf", fontsize=14, y=1.01)
    _savefig(fig, Path(out_path), "Comparison summary")
