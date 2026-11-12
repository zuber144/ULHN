"""
navigation/policy.py
====================
Custom MLP policy configuration for PPO.

We use Stable-Baselines3's built-in MlpPolicy but expose the network
architecture as configurable parameters so they can be tuned in config.yaml
without code changes.
"""

from __future__ import annotations

from typing import List, Optional, Type

import torch
import torch.nn as nn


# Default network architecture (overrideable from config.yaml)
DEFAULT_NET_ARCH = [256, 256]


def get_policy_kwargs(
    net_arch: Optional[List[int]] = None,
    activation_fn: str = "tanh",
) -> dict:
    """
    Build the policy_kwargs dict for SB3's MlpPolicy.

    Parameters
    ----------
    net_arch:
        Hidden layer sizes shared between policy and value networks.
    activation_fn:
        One of 'tanh', 'relu', 'elu'.

    Returns
    -------
    dict compatible with SB3 MlpPolicy's policy_kwargs argument.
    """
    arch = net_arch or DEFAULT_NET_ARCH

    _activations: dict = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu":  nn.ELU,
    }
    fn: Type[nn.Module] = _activations.get(activation_fn.lower(), nn.Tanh)

    return {
        "net_arch": arch,
        "activation_fn": fn,
    }
