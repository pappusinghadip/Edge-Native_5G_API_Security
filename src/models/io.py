"""Shared helpers for loading processed data and resolving model artifact paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import resolve_path


@dataclass(frozen=True)
class ModelArtifactPaths:
    train_split_path: Path
    val_split_path: Path
    test_split_path: Path
    best_model_path: Path
    final_model_path: Path
    training_log_path: Path
    tensorboard_dir: Path
    metrics_path: Path
    summary_path: Path
    confusion_matrix_figure_path: Path
    roc_figure_path: Path
    training_curves_figure_path: Path
    tflite_model_path: Path
    latency_metrics_path: Path


def load_processed_split(split_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a processed split archive into arrays."""
    with np.load(split_path) as data:
        return data["X"], data["y"], data["indices"]


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels into one-hot encoded arrays."""
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TensorFlow is required for label encoding.") from exc

    return tf.keras.utils.to_categorical(labels, num_classes=num_classes)


def resolve_model_artifact_paths(paths_config: dict[str, Any], model_config: dict[str, Any]) -> ModelArtifactPaths:
    """Resolve all filesystem paths used by Phase 2."""
    processed_dir = resolve_path(paths_config["data"]["processed"])
    models_dir = resolve_path(paths_config["results"]["models"])
    metrics_dir = resolve_path(paths_config["results"]["metrics"])
    figures_dir = resolve_path(paths_config["results"]["figures"])
    tensorboard_root = resolve_path(paths_config["results"]["tensorboard"])

    training_cfg = model_config["training"]
    evaluation_cfg = model_config["evaluation"]
    export_cfg = model_config["export"]

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_root.mkdir(parents=True, exist_ok=True)

    return ModelArtifactPaths(
        train_split_path=processed_dir / "train.npz",
        val_split_path=processed_dir / "val.npz",
        test_split_path=processed_dir / "test.npz",
        best_model_path=models_dir / training_cfg["checkpoint_name"],
        final_model_path=models_dir / training_cfg["final_model_name"],
        training_log_path=metrics_dir / training_cfg["training_log_name"],
        tensorboard_dir=tensorboard_root / training_cfg["tensorboard_subdir"],
        metrics_path=metrics_dir / evaluation_cfg["metrics_name"],
        summary_path=metrics_dir / evaluation_cfg["summary_name"],
        confusion_matrix_figure_path=figures_dir / evaluation_cfg["confusion_matrix_figure"],
        roc_figure_path=figures_dir / evaluation_cfg["roc_figure"],
        training_curves_figure_path=figures_dir / evaluation_cfg["training_curves_figure"],
        tflite_model_path=models_dir / export_cfg["tflite_name"],
        latency_metrics_path=metrics_dir / export_cfg["latency_metrics_name"],
    )


def save_model_summary(model: Any, summary_path: Path) -> None:
    """Persist the textual Keras model summary."""
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
