"""
watchdog/train_watchdog.py
==========================
Training pipeline for the Autoencoder watchdog.

Steps:
    1. Load normal LiDAR dataset
    2. Build Autoencoder
    3. Train on dataset
    4. Compute anomaly scores on held-out validation set
    5. Compute and save threshold statistics
    6. Save model
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from watchdog.autoencoder import (
    LiDARAutoencoder,
    train_autoencoder,
    save_autoencoder,
)
from watchdog.dataset import load_dataset
from environment.lidar import select_forward_sector
from watchdog.threshold import (
    ThresholdStats,
    compute_threshold_stats,
    save_threshold_stats,
)

logger = logging.getLogger(__name__)


def train_watchdog_pipeline(config: dict) -> Tuple[LiDARAutoencoder, ThresholdStats]:
    """
    Full watchdog training pipeline.

    Parameters
    ----------
    config:
        Full config dict from config.yaml.

    Returns
    -------
    (trained_model, threshold_stats)
    """
    wd_cfg = config.get("watchdog", {})
    lidar_cfg = config.get("lidar", {})

    hidden_dim  = wd_cfg.get("hidden_dim", 16)
    latent_dim  = wd_cfg.get("latent_dim", 8)
    lr          = wd_cfg.get("learning_rate", 1e-3)
    batch_size  = wd_cfg.get("batch_size", 256)
    epochs      = wd_cfg.get("epochs", 50)
    dataset_dir = wd_cfg.get("dataset_dir", "data/normal_lidar")
    model_dir   = Path(wd_cfg.get("model_dir", "models/autoencoder"))
    percentile  = wd_cfg.get("threshold_percentile", 95.0)
    forward_fraction = wd_cfg.get("forward_sector_fraction", 0.5)

    model_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data = load_dataset(dataset_dir)
    logger.info("Training Autoencoder on %d observations", len(data))
    forward_data = select_forward_sector(data, forward_fraction)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model = LiDARAutoencoder(
        input_dim=forward_data.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )

    # Train
    train_losses, val_losses = train_autoencoder(
        model=model,
        data=forward_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_split=0.1,
        device=device,
    )

    # Compute anomaly scores on validation portion for threshold
    logger.info("Computing threshold statistics on validation data …")
    model.eval()
    model_cpu = model.to("cpu")
    x_val = torch.tensor(
        forward_data[-max(100, len(forward_data) // 10):], dtype=torch.float32
    )
    scores = model_cpu.anomaly_score(x_val).numpy()
    stats = compute_threshold_stats(scores)

    threshold = stats.p95 if percentile == 95.0 else float(np.percentile(scores, percentile))
    logger.info(
        "Threshold stats: mean=%.6f, p95=%.6f, p99=%.6f",
        stats.mean, stats.p95, stats.p99,
    )

    # Save
    save_autoencoder(model, str(model_dir / "autoencoder.pt"))
    save_threshold_stats(stats, str(model_dir / "threshold_stats.json"))

    # Save training losses
    import json
    with (model_dir / "training_losses.json").open("w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
        }, f)

    logger.info("Watchdog training complete.")
    return model, stats
