"""Non-learning baseline: a static rate/threshold detector.

This is the "simple non-learning edge system" comparator required by the
validation phase (Task 3.1). It emulates a classic rate-limiter / WAF rule:
flag a flow as malicious when its request rate or SYN count exceeds a fixed
threshold. The threshold is set from benign-traffic statistics (a configured
percentile), not learned — so it is a genuine non-ML baseline.

Feature indices (from configs/model.yaml feature order):
  3 = Rate, 5 = syn_count  (both are elevated for brute-force / flood traffic)

Run:  python -m src.models.baseline --paths configs/paths.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.evaluation.metrics import false_positive_rate
from src.models.io import load_processed_split
from src.utils.config import load_yaml, resolve_path

LOGGER = logging.getLogger(__name__)

RATE_IDX = 3
SYN_IDX = 5
BENIGN_PERCENTILE = 99.0  # fixed operating point; a real rate-limiter would tune this per deployment


def rule_score(X: np.ndarray) -> np.ndarray:
    """Heuristic anomaly score = max(scaled rate, scaled syn_count) per flow."""
    flat = X.reshape(X.shape[0], -1)
    return np.maximum(flat[:, RATE_IDX], flat[:, SYN_IDX])


def evaluate_baseline(processed_dir: Path) -> dict:
    _, y_train, _ = load_processed_split(processed_dir / "train.npz")
    X_train, _, _ = load_processed_split(processed_dir / "train.npz")
    X_test, y_test, _ = load_processed_split(processed_dir / "test.npz")

    train_scores = rule_score(X_train)
    test_scores = rule_score(X_test)

    # Static threshold from benign traffic only (configuration, not learning)
    benign_scores = train_scores[y_train == 0]
    threshold = float(np.percentile(benign_scores, BENIGN_PERCENTILE))
    preds = (test_scores > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", pos_label=1, zero_division=0
    )
    tn = int(np.sum((preds == 0) & (y_test == 0)))
    fp = int(np.sum((preds == 1) & (y_test == 0)))
    try:
        auc = float(roc_auc_score(y_test, test_scores))
    except ValueError:
        auc = float("nan")

    return {
        "model": "static_rate_limiter",
        "rule": f"flag if max(Rate, syn_count) > p{BENIGN_PERCENTILE:g}(benign)",
        "threshold": threshold,
        "accuracy": float(np.mean(preds == y_test)),
        "precision_binary": float(precision),
        "recall_binary": float(recall),
        "f1_binary": float(f1),
        "auc_roc": auc,
        "false_positive_rate": float(false_positive_rate(fp, tn)),
    }


def main() -> None:
    from src.utils.logger import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    args = parser.parse_args()
    paths_config = load_yaml(args.paths)
    processed_dir = resolve_path(paths_config["data"]["processed"])
    metrics = evaluate_baseline(processed_dir)

    out = resolve_path(paths_config["results"]["metrics"]) / "baseline_ratelimiter.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Non-learning baseline: AUC=%.4f F1=%.4f FPR=%.4f -> %s",
                metrics["auc_roc"], metrics["f1_binary"], metrics["false_positive_rate"], out)


if __name__ == "__main__":
    main()
