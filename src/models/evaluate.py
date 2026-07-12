"""Evaluation entrypoint for the centralized 1D-CNN baseline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.metrics import false_positive_rate
from src.models.io import load_processed_split, resolve_model_artifact_paths, save_model_summary
from src.utils.config import load_yaml
from src.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationArtifacts:
    metrics_path: Path
    summary_path: Path
    confusion_matrix_figure_path: Path
    roc_figure_path: Path
    training_curves_figure_path: Path


def select_decision_threshold(
    y_true: Any,
    y_score: Any,
    threshold_config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Select a malicious-class threshold from validation probabilities."""
    if not threshold_config.get("enabled", True):
        return 0.5, {"f1_binary": float(f1_score(y_true, y_score >= 0.5, zero_division=0))}

    thresholds = [
        float(value)
        for value in np.linspace(
            float(threshold_config.get("min", 0.01)),
            float(threshold_config.get("max", 0.99)),
            int(threshold_config.get("steps", 99)),
        )
    ]

    best_threshold = 0.5
    best_stats = {"f1_binary": -1.0, "precision_binary": 0.0, "recall_binary": 0.0}
    for threshold in thresholds:
        predictions = (y_score >= threshold).astype(int)
        precision, recall, f1_binary, _ = precision_recall_fscore_support(
            y_true,
            predictions,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        stats = {
            "f1_binary": float(f1_binary),
            "precision_binary": float(precision),
            "recall_binary": float(recall),
        }
        if (
            stats["f1_binary"] > best_stats["f1_binary"]
            or (
                stats["f1_binary"] == best_stats["f1_binary"]
                and stats["recall_binary"] > best_stats["recall_binary"]
            )
        ):
            best_threshold = threshold
            best_stats = stats
    return best_threshold, best_stats


def plot_confusion_matrix_figure(matrix: Any, output_path: Path) -> None:
    """Save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Benign", "Malicious"])
    ax.set_yticklabels(["Benign", "Malicious"], rotation=0)
    ax.set_title("Centralized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_figure(y_true: Any, y_score: Any, output_path: Path) -> float:
    """Save the ROC curve and return the AUC."""
    fpr_values, tpr_values, _ = roc_curve(y_true, y_score)
    auc_value = float(auc(fpr_values, tpr_values))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr_values, tpr_values, label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Centralized ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return auc_value


def plot_training_curves(training_log_path: Path, output_path: Path) -> None:
    """Save loss and accuracy training curves from the CSV logger output."""
    history = pd.read_csv(training_log_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history["epoch"], history["loss"], label="train_loss")
    if "val_loss" in history.columns:
        axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["accuracy"], label="train_accuracy")
    if "val_accuracy" in history.columns:
        axes[1].plot(history["epoch"], history["val_accuracy"], label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_centralized_model(model_config: dict[str, Any], paths_config: dict[str, Any]) -> EvaluationArtifacts:
    """Evaluate the centralized model on the held-out test split and save artifacts."""
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TensorFlow is required for model evaluation.") from exc

    paths = resolve_model_artifact_paths(paths_config, model_config)
    if not paths.best_model_path.exists():
        raise FileNotFoundError(f"Centralized model not found at {paths.best_model_path}")

    from src.models.cnn import focal_loss

    model = tf.keras.models.load_model(
        paths.best_model_path,
        custom_objects={"focal_loss": focal_loss()},
    )
    X_val, y_val, _ = load_processed_split(paths.val_split_path)
    X_test, y_test, _ = load_processed_split(paths.test_split_path)
    val_probabilities = model.predict(
        X_val,
        batch_size=int(model_config["training"]["batch_size"]),
        verbose=0,
    )
    probabilities = model.predict(
        X_test,
        batch_size=int(model_config["training"]["batch_size"]),
        verbose=0,
    )
    threshold, threshold_stats = select_decision_threshold(
        y_val,
        val_probabilities[:, 1],
        model_config["evaluation"].get("threshold_search", {}),
    )
    predictions = (probabilities[:, 1] >= threshold).astype(int)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    auc_value = float(roc_auc_score(y_test, probabilities[:, 1]))
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_binary": float(precision_binary),
        "recall_binary": float(recall_binary),
        "f1_binary": float(f1_binary),
        "auc_roc": auc_value,
        "false_positive_rate": float(false_positive_rate(int(fp), int(tn))),
        "decision_threshold": float(threshold),
        "validation_threshold_stats": threshold_stats,
        "confusion_matrix": matrix.tolist(),
        "support": {
            "benign": int((y_test == 0).sum()),
            "malicious": int((y_test == 1).sum()),
        },
    }

    paths.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_model_summary(model, paths.summary_path)
    plot_confusion_matrix_figure(matrix, paths.confusion_matrix_figure_path)
    plot_roc_figure(y_test, probabilities[:, 1], paths.roc_figure_path)
    plot_training_curves(paths.training_log_path, paths.training_curves_figure_path)

    return EvaluationArtifacts(
        metrics_path=paths.metrics_path,
        summary_path=paths.summary_path,
        confusion_matrix_figure_path=paths.confusion_matrix_figure_path,
        roc_figure_path=paths.roc_figure_path,
        training_curves_figure_path=paths.training_curves_figure_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/model.yaml")
    parser.add_argument("--paths", required=True, help="Path to configs/paths.yaml")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    model_config = load_yaml(args.config)
    paths_config = load_yaml(args.paths)
    artifacts = evaluate_centralized_model(model_config, paths_config)
    LOGGER.info("Saved metrics to %s", artifacts.metrics_path)
    LOGGER.info("Saved model summary to %s", artifacts.summary_path)
    LOGGER.info("Saved confusion matrix figure to %s", artifacts.confusion_matrix_figure_path)
    LOGGER.info("Saved ROC figure to %s", artifacts.roc_figure_path)
    LOGGER.info("Saved training curves to %s", artifacts.training_curves_figure_path)


if __name__ == "__main__":
    main()
