"""tests/test_watchdog.py — Unit tests for Autoencoder watchdog."""

from __future__ import annotations

import numpy as np
import pytest
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _load(name, relpath):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load all required modules
_load("watchdog.autoencoder",     "watchdog/autoencoder.py")
_load("watchdog.threshold",       "watchdog/threshold.py")
_load("watchdog.anomaly_detector","watchdog/anomaly_detector.py")

import torch
from watchdog.autoencoder import LiDARAutoencoder, train_autoencoder
from watchdog.threshold import compute_threshold_stats, select_threshold
from watchdog.anomaly_detector import AnomalyDetector


def _make_model(n: int = 36) -> LiDARAutoencoder:
    return LiDARAutoencoder(input_dim=n, hidden_dim=16, latent_dim=8)


class TestLiDARAutoencoder:
    def test_forward_shape(self):
        model = _make_model()
        x = torch.rand(4, 36)
        y = model(x)
        assert y.shape == (4, 36)

    def test_output_in_range(self):
        model = _make_model()
        x = torch.rand(10, 36)
        y = model(x)
        assert torch.all(y >= 0.0) and torch.all(y <= 1.0)

    def test_anomaly_score_shape(self):
        model = _make_model()
        x = torch.rand(5, 36)
        scores = model.anomaly_score(x)
        assert scores.shape == (5,)

    def test_anomaly_score_numpy(self):
        model = _make_model()
        obs = np.random.rand(36).astype(np.float32)
        score = model.anomaly_score_numpy(obs)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_train_reduces_loss(self):
        """After training, reconstruction loss should decrease."""
        model = _make_model()
        data = np.random.rand(500, 36).astype(np.float32)
        train_losses, _ = train_autoencoder(
            model, data, epochs=10, batch_size=64, lr=1e-3,
            val_split=0.1, device="cpu",
        )
        assert train_losses[-1] < train_losses[0], \
            f"Loss did not decrease: {train_losses[0]:.4f} → {train_losses[-1]:.4f}"

    def test_normal_vs_ood_scores(self):
        """Normal data should have lower scores than highly corrupted data."""
        model = _make_model()
        # Train on uniform data
        normal = np.random.uniform(0.7, 0.9, (300, 36)).astype(np.float32)
        train_autoencoder(model, normal, epochs=20, batch_size=64, lr=1e-3,
                          val_split=0.1, device="cpu")
        model.eval()

        normal_scores = np.array([model.anomaly_score_numpy(s) for s in normal[:50]])
        # OOD: zeros (very different from training distribution)
        ood = np.zeros((50, 36), dtype=np.float32)
        ood_scores = np.array([model.anomaly_score_numpy(s) for s in ood])

        assert np.mean(ood_scores) > np.mean(normal_scores), \
            "OOD should have higher scores than normal"


class TestThreshold:
    def test_percentile_selection(self):
        scores = np.random.exponential(0.01, 1000)
        stats = compute_threshold_stats(scores)
        t95 = select_threshold(stats, 95.0)
        assert t95 > 0.0
        assert t95 >= stats.p90

    def test_stats_fields(self):
        scores = np.linspace(0.001, 0.1, 1000)
        stats = compute_threshold_stats(scores)
        assert stats.n_samples == 1000
        assert stats.p50 < stats.p95 < stats.p99


class TestAnomalyDetector:
    def setup_method(self):
        self.model = _make_model(18)
        # Minimal training
        data = np.random.rand(200, 18).astype(np.float32)
        train_autoencoder(self.model, data, epochs=5, batch_size=64, lr=1e-3,
                          val_split=0.1, device="cpu")
        self.model.eval()
        self.detector = AnomalyDetector(self.model, threshold=0.05, smoothing_window=3)

    def test_update_returns_float(self):
        obs = np.random.rand(36).astype(np.float32)
        score = self.detector.update(obs)
        assert isinstance(score, float)

    def test_reset_clears_buffer(self):
        obs = np.ones(36, dtype=np.float32)
        self.detector.update(obs)
        self.detector.reset()
        assert self.detector.smoothed_score == 0.0

    def test_is_ood_respects_threshold(self):
        # Force a very high anomaly by passing zeros to a model trained on 0.5-ish
        self.detector.threshold = 1e-10  # extremely low threshold
        obs = np.zeros(36, dtype=np.float32)
        # Update enough times to fill smoothing buffer
        for _ in range(5):
            self.detector.update(obs)
        assert self.detector.is_ood()

    def test_rear_sector_does_not_change_runtime_score(self):
        class RecordingModel:
            def anomaly_score_numpy(self, obs):
                self.last_obs = obs.copy()
                return float(np.mean(obs))

        model = RecordingModel()
        detector = AnomalyDetector(model, threshold=0.5, smoothing_window=1)
        observation = np.ones(36, dtype=np.float32)
        rear_changed = observation.copy()
        rear_changed[:9] = 0.0
        rear_changed[27:] = 0.0

        assert detector.score(rear_changed) == detector.score(observation)
        assert model.last_obs.shape == (18,)

        forward_changed = observation.copy()
        forward_changed[18] = 0.0
        assert detector.score(forward_changed) < detector.score(observation)
