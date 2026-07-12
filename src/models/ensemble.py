"""Hybrid CNN + Gradient-Boosting ensemble.

The revised plan calls for a hybrid ML ensemble rather than a single model.
Tree-based models are known to be strong on tabular/flow features, while the
1D-CNN captures local cross-feature patterns; averaging their probabilities
gives a detector that is at least as good as the better single model.

Components:
  * 1D-CNN   — the trained centralized model (results/models/centralized_best.h5)
  * HistGBM  — sklearn HistGradientBoostingClassifier on the same 10 features
  * Ensemble — weighted average of the two probability outputs (weight tuned on val AUC)

Run:  python -m src.models.ensemble --model configs/model.yaml --paths configs/paths.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.evaluation.metrics import false_positive_rate
from src.models.io import load_processed_split
from src.utils.config import load_yaml, resolve_path
from src.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)


def _flat(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def best_threshold(y: np.ndarray, score: np.ndarray, steps: int = 199) -> float:
    from sklearn.metrics import f1_score

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, steps):
        f1 = f1_score(y, (score >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t


def scored_metrics(name: str, y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    preds = (score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", pos_label=1, zero_division=0
    )
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    return {
        "model": name,
        "auc_roc": float(roc_auc_score(y, score)),
        "precision_binary": float(precision),
        "recall_binary": float(recall),
        "f1_binary": float(f1),
        "false_positive_rate": float(false_positive_rate(fp, tn)),
        "threshold": float(threshold),
    }


def run_ensemble(model_config: dict, paths_config: dict) -> dict:
    import tensorflow as tf

    from src.models.cnn import focal_loss

    set_global_seed(int(model_config.get("random_seed", 42)))
    processed = resolve_path(paths_config["data"]["processed"])
    models_dir = resolve_path(paths_config["results"]["models"])

    X_train, y_train, _ = load_processed_split(processed / "train.npz")
    X_val, y_val, _ = load_processed_split(processed / "val.npz")
    X_test, y_test, _ = load_processed_split(processed / "test.npz")

    # --- Gradient boosting on flat features ---
    LOGGER.info("Training HistGradientBoosting on %d samples...", len(y_train))
    gbm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42)
    gbm.fit(_flat(X_train), y_train)
    gbm_val = gbm.predict_proba(_flat(X_val))[:, 1]
    gbm_test = gbm.predict_proba(_flat(X_test))[:, 1]

    # --- CNN probabilities ---
    cnn = tf.keras.models.load_model(
        models_dir / model_config["training"]["checkpoint_name"],
        custom_objects={"focal_loss": focal_loss()},
    )
    cnn_val = cnn.predict(X_val, batch_size=1024, verbose=0)[:, 1]
    cnn_test = cnn.predict(X_test, batch_size=1024, verbose=0)[:, 1]

    # --- Tune ensemble weight on validation AUC ---
    best_w, best_auc = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, 21):
        auc = roc_auc_score(y_val, w * cnn_val + (1 - w) * gbm_val)
        if auc > best_auc:
            best_auc, best_w = float(auc), float(w)
    ens_val = best_w * cnn_val + (1 - best_w) * gbm_val
    ens_test = best_w * cnn_test + (1 - best_w) * gbm_test

    results = {
        "cnn": scored_metrics("1D-CNN", y_test, cnn_test, best_threshold(y_val, cnn_val)),
        "gbm": scored_metrics("HistGradientBoosting", y_test, gbm_test, best_threshold(y_val, gbm_val)),
        "ensemble": scored_metrics("CNN+GBM ensemble", y_test, ens_test, best_threshold(y_val, ens_val)),
        "ensemble_weight_cnn": best_w,
    }
    return results


def main() -> None:
    from src.utils.logger import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--paths", required=True)
    args = parser.parse_args()
    model_config = load_yaml(args.model)
    paths_config = load_yaml(args.paths)

    results = run_ensemble(model_config, paths_config)
    out = resolve_path(paths_config["results"]["metrics"]) / "ensemble_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for key in ("cnn", "gbm", "ensemble"):
        m = results[key]
        LOGGER.info("%-20s AUC=%.4f F1=%.4f FPR=%.4f", m["model"], m["auc_roc"], m["f1_binary"], m["false_positive_rate"])
    LOGGER.info("ensemble weight (CNN)=%.2f -> %s", results["ensemble_weight_cnn"], out)


if __name__ == "__main__":
    main()
