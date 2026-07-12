import numpy as np

from src.fl.strategy import is_update_within_norm, update_norm


def test_update_norm_returns_positive_value() -> None:
    weights = [np.array([3.0, 4.0])]
    assert update_norm(weights) == 5.0


def test_clipping_accepts_small_updates() -> None:
    weights = [np.array([1.0, 2.0])]
    assert is_update_within_norm(weights, max_grad_norm=5.0)
