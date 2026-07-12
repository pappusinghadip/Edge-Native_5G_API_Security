"""Shared metric utilities for model and FL evaluations."""

from __future__ import annotations


def false_positive_rate(false_positives: int, true_negatives: int) -> float:
    """Compute the false positive rate from confusion matrix counts."""
    denominator = false_positives + true_negatives
    if denominator == 0:
        return 0.0
    return false_positives / denominator
