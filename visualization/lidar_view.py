"""
visualization/lidar_view.py
============================
LiDAR polar/bar visualisation helpers used by the dashboard.
"""

from __future__ import annotations

import math
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def lidar_polar_figure(
    lidar_obs: np.ndarray,
    num_rays: int = 36,
    fov_deg: float = 360.0,
    anomaly_score: Optional[float] = None,
    threshold: Optional[float] = None,
    figsize: tuple = (5, 5),
) -> plt.Figure:
    """
    Create a polar plot of LiDAR observations.

    Returns a Matplotlib Figure (caller is responsible for closing).
    """
    fov_rad = math.radians(fov_deg)
    if num_rays == 1:
        angles = np.array([0.0])
    else:
        angles = np.linspace(
            -fov_rad / 2.0, fov_rad / 2.0, num_rays,
            endpoint=(fov_deg < 360.0),
        )

    distances = lidar_obs.copy()

    fig = plt.figure(figsize=figsize, facecolor="#0f0f1e")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("#1a1a2e")

    is_ood = (anomaly_score is not None and threshold is not None and
              anomaly_score > threshold)
    color = "#ff4444" if is_ood else "#00ccff"

    ax.bar(
        angles, distances,
        width=fov_rad / num_rays * 0.9,
        color=color, alpha=0.6, bottom=0.0,
    )
    ax.set_ylim(0, 1.1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(labelcolor="#aaa")
    ax.set_facecolor("#1a1a2e")

    title = "LiDAR Scan"
    if anomaly_score is not None:
        status = "OOD" if is_ood else "NORMAL"
        title += f"  [{status}  score={anomaly_score:.4f}]"
    ax.set_title(title, color="#ccf", fontsize=10, pad=15)

    fig.patch.set_facecolor("#0f0f1e")
    return fig


def lidar_bar_figure(
    lidar_obs: np.ndarray,
    figsize: tuple = (10, 2),
) -> plt.Figure:
    """Simple bar chart of LiDAR ray values (0→blocked, 1→clear)."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f0f1e")
    ax.set_facecolor("#1a1a2e")
    x = np.arange(len(lidar_obs))
    ax.bar(x, lidar_obs, color="#00ccff", alpha=0.7, width=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Ray index", color="#aaa")
    ax.set_ylabel("Normalised dist.", color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.set_title("LiDAR Ray Distances", color="#ccf", fontsize=11)
    fig.tight_layout()
    return fig
