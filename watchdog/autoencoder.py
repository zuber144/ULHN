"""
watchdog/autoencoder.py
=======================
Lightweight PyTorch Autoencoder for LiDAR OOD detection.

Architecture (defaults from config.yaml):
    Input: 36  →  16  →  8  →  16  →  36
    Loss:  Mean Squared Error (MSE)

The Autoencoder learns to reconstruct *normal* LiDAR observations. When
presented with OOD inputs (corrupted / abnormal LiDAR), the reconstruction
error increases above a statistically derived threshold.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class LiDARAutoencoder(nn.Module):
    """
    Fully-connected autoencoder for normalised LiDAR observations.

    Parameters
    ----------
    input_dim:
        Number of LiDAR rays (default 36).
    hidden_dim:
        First hidden layer size (default 16).
    latent_dim:
        Bottleneck dimension (default 8).
    """

    def __init__(
        self,
        input_dim: int = 36,
        hidden_dim: int = 16,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: input → hidden → latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        # Decoder: latent → hidden → output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),   # output in [0, 1] matching normalised LiDAR
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        latent = self.encoder(x)
        return self.decoder(latent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation."""
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(x)

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample MSE reconstruction error.

        Parameters
        ----------
        x : torch.Tensor of shape (N, input_dim) or (input_dim,)

        Returns
        -------
        torch.Tensor of shape (N,) — per-sample anomaly scores.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            recon = self.forward(x)
            scores = torch.mean((x - recon) ** 2, dim=1)
        return scores

    def anomaly_score_numpy(self, obs: np.ndarray) -> float:
        """
        Compute anomaly score for a single NumPy LiDAR observation.

        Parameters
        ----------
        obs : np.ndarray of shape (input_dim,)

        Returns
        -------
        float — MSE reconstruction error.
        """
        device = next(self.parameters()).device
        t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        score = self.anomaly_score(t)
        return float(score.item())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_autoencoder(
    model: LiDARAutoencoder,
    data: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device: str = "cpu",
) -> Tuple[List[float], List[float]]:
    """
    Train the Autoencoder on normal LiDAR data.

    Parameters
    ----------
    model:
        Uninitialised or partially trained LiDARAutoencoder.
    data:
        Normal LiDAR dataset, shape (N, input_dim), values in [0, 1].
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Adam learning rate.
    val_split:
        Fraction of data to hold out for validation.
    device:
        'cpu' or 'cuda'.

    Returns
    -------
    train_losses, val_losses : lists of per-epoch MSE.
    """
    model = model.to(device)
    n = len(data)
    n_val = max(1, int(n * val_split))
    idx = np.random.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_train = torch.tensor(data[train_idx], dtype=torch.float32)
    x_val   = torch.tensor(data[val_idx],   dtype=torch.float32)

    train_ds = TensorDataset(x_train)
    val_ds   = TensorDataset(x_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses: List[float] = []
    val_losses:   List[float] = []

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_dl:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        train_losses.append(epoch_loss / len(x_train))

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_dl:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += criterion(recon, batch).item() * len(batch)
        val_losses.append(val_loss / len(x_val))

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Epoch %3d/%d | train_loss=%.6f | val_loss=%.6f",
                epoch + 1, epochs, train_losses[-1], val_losses[-1],
            )

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_autoencoder(model: LiDARAutoencoder, path: str) -> None:
    """Save model weights and architecture config."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim":  model.input_dim,
        "hidden_dim": model.hidden_dim,
        "latent_dim": model.latent_dim,
    }, str(out))
    logger.info("Autoencoder saved to %s", out)


def load_autoencoder(path: str, device: str = "cpu") -> LiDARAutoencoder:
    """Load a saved autoencoder."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {p.resolve()}")
    ckpt = torch.load(str(p), map_location=device)
    model = LiDARAutoencoder(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        latent_dim=ckpt["latent_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    logger.info("Autoencoder loaded from %s", p)
    return model
