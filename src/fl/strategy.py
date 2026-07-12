"""SafeFedAvg strategy helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def update_norm(weights: list[np.ndarray]) -> float:
    """Compute a simple aggregated L2 norm over a list of tensors."""
    return float(sum(np.linalg.norm(weight) for weight in weights))


def is_update_within_norm(weights: list[np.ndarray], max_grad_norm: float) -> bool:
    """Return True when an update stays within the clipping threshold."""
    return update_norm(weights) <= max_grad_norm


def build_safe_fedavg(max_grad_norm: float, **kwargs: Any) -> Any:
    """Build a Flower FedAvg strategy with later clipping support."""
    try:
        import flwr as fl
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Flower is required to build the FL strategy.") from exc

    return fl.server.strategy.FedAvg(**kwargs)
