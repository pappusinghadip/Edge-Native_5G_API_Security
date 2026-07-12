"""Centralized 1D-CNN training entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.models.cnn import build_model_from_config
from src.models.io import load_processed_split, one_hot_encode, resolve_model_artifact_paths
from src.utils.config import load_yaml
from src.utils.logger import configure_logging
from src.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingArtifacts:
    best_model_path: Path
    final_model_path: Path
    training_log_path: Path
    tensorboard_dir: Path
    training_metadata_path: Path
    epochs_completed: int


def build_callbacks(model_config: dict[str, Any], paths: Any) -> list[Any]:
    """Create the standard centralized training callbacks."""
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TensorFlow is required for model training.") from exc

    early_cfg = model_config["training"]["early_stopping"]
    monitor = early_cfg["monitor"]
    mode = str(early_cfg.get("mode", "auto"))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=int(early_cfg["patience"]),
            restore_best_weights=bool(early_cfg["restore_best_weights"]),
            mode=mode,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(paths.best_model_path),
            monitor=monitor,
            save_best_only=True,
            mode=mode,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(paths.tensorboard_dir)),
        tf.keras.callbacks.CSVLogger(str(paths.training_log_path)),
    ]

    lr_cfg = model_config["training"].get("reduce_lr")
    if lr_cfg:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=str(lr_cfg.get("monitor", "val_loss")),
                factor=float(lr_cfg.get("factor", 0.5)),
                patience=int(lr_cfg.get("patience", 5)),
                min_lr=float(lr_cfg.get("min_lr", 1e-5)),
                mode=str(lr_cfg.get("mode", "auto")),
                verbose=1,
            )
        )

    return callbacks


def load_training_splits(paths: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load processed train and validation splits with integer labels."""
    X_train, y_train, _ = load_processed_split(paths.train_split_path)
    X_val, y_val, _ = load_processed_split(paths.val_split_path)
    return X_train, y_train, X_val, y_val


def oversample_minority_class(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Oversample minority classes using SMOTE with random oversampling fallback.

    SMOTE generates synthetic samples by interpolating between existing minority
    examples in feature space. This produces far more diverse training data than
    naive duplication, which just memorises the same examples.
    """
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")

    original_shape = X.shape
    # Flatten from (N, timesteps, channels) to (N, features) for SMOTE
    X_flat = X.reshape(X.shape[0], -1)

    classes, counts = np.unique(y, return_counts=True)
    majority_count = int(max(counts))
    target_count = int(round(majority_count * target_ratio))
    sampling_strategy = {int(c): max(target_count, int(cnt)) for c, cnt in zip(classes, counts, strict=True)}

    try:
        from imblearn.over_sampling import SMOTE

        minority_count = int(min(counts))
        k_neighbors = min(5, minority_count - 1)
        if k_neighbors < 1:
            raise ValueError("Not enough minority samples for SMOTE")

        smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=seed)
        X_resampled, y_resampled = smote.fit_resample(X_flat, y)
        LOGGER.info("SMOTE resampling succeeded (k_neighbors=%d)", k_neighbors)
    except (ImportError, ValueError) as exc:
        LOGGER.warning("SMOTE unavailable (%s), falling back to random oversampling", exc)
        rng = np.random.default_rng(seed)
        sampled_indices: list[np.ndarray] = []
        for class_id in classes:
            class_indices = np.flatnonzero(y == class_id)
            original_count = len(class_indices)
            if original_count >= target_count:
                chosen = class_indices.copy()
            else:
                extra = rng.choice(class_indices, size=target_count - original_count, replace=True)
                chosen = np.concatenate([class_indices, extra])
            sampled_indices.append(chosen)
        merged_indices = np.concatenate(sampled_indices)
        rng.shuffle(merged_indices)
        X_resampled = X_flat[merged_indices]
        y_resampled = y[merged_indices]

    # Restore CNN shape
    new_shape = (X_resampled.shape[0],) + original_shape[1:]
    X_resampled = X_resampled.reshape(new_shape).astype(np.float32)

    resampled_counts = {
        str(int(c)): int(cnt)
        for c, cnt in zip(*np.unique(y_resampled, return_counts=True), strict=True)
    }
    return X_resampled, y_resampled, resampled_counts


def compute_sample_weights(labels: np.ndarray, strategy: str) -> tuple[np.ndarray, dict[str, float]]:
    """Compute per-sample weights from integer labels."""
    if strategy != "balanced":
        weights = np.ones_like(labels, dtype=np.float32)
        return weights, {"0": 1.0, "1": 1.0}

    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    class_weights = {
        int(class_id): float(total / (len(classes) * count))
        for class_id, count in zip(classes, counts, strict=True)
    }
    sample_weights = np.asarray([class_weights[int(label)] for label in labels], dtype=np.float32)
    return sample_weights, {str(class_id): weight for class_id, weight in class_weights.items()}


def train_centralized_model(model_config: dict[str, Any], paths_config: dict[str, Any]) -> TrainingArtifacts:
    """Train the centralized 1D-CNN baseline and persist the best and final models."""
    set_global_seed(int(model_config["random_seed"]))
    paths = resolve_model_artifact_paths(paths_config, model_config)
    paths.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    num_classes = int(model_config["model"]["num_classes"])
    X_train, y_train_labels, X_val, y_val_labels = load_training_splits(paths)
    training_cfg = model_config["training"]

    resampled_counts = {str(int(class_id)): int(count) for class_id, count in zip(*np.unique(y_train_labels, return_counts=True), strict=True)}
    if training_cfg.get("resampling_strategy", "none") == "oversample_minority":
        X_train, y_train_labels, resampled_counts = oversample_minority_class(
            X_train,
            y_train_labels,
            target_ratio=float(training_cfg.get("oversampling_target_ratio", 1.0)),
            seed=int(model_config["random_seed"]),
        )

    y_train = one_hot_encode(y_train_labels, num_classes)
    y_val = one_hot_encode(y_val_labels, num_classes)
    sample_weights, class_weights = compute_sample_weights(
        y_train_labels,
        strategy=str(training_cfg.get("class_weighting", "none")),
    )

    model = build_model_from_config(model_config)
    callbacks = build_callbacks(model_config, paths)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(training_cfg["epochs"]),
        batch_size=int(training_cfg["batch_size"]),
        callbacks=callbacks,
        sample_weight=sample_weights if str(training_cfg.get("class_weighting", "none")) != "none" else None,
        verbose=2,
    )
    model.save(paths.final_model_path)
    training_metadata_path = paths.training_log_path.with_name("centralized_training_metadata.json")
    training_metadata = {
        "class_weighting": training_cfg.get("class_weighting", "none"),
        "resampling_strategy": training_cfg.get("resampling_strategy", "none"),
        "oversampling_target_ratio": float(training_cfg.get("oversampling_target_ratio", 1.0)),
        "class_weights": class_weights,
        "training_counts_after_resampling": resampled_counts,
        "epochs_completed": len(history.history.get("loss", [])),
    }
    training_metadata_path.write_text(json.dumps(training_metadata, indent=2), encoding="utf-8")

    return TrainingArtifacts(
        best_model_path=paths.best_model_path,
        final_model_path=paths.final_model_path,
        training_log_path=paths.training_log_path,
        tensorboard_dir=paths.tensorboard_dir,
        training_metadata_path=training_metadata_path,
        epochs_completed=len(history.history.get("loss", [])),
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
    artifacts = train_centralized_model(model_config, paths_config)
    LOGGER.info("Saved best model to %s", artifacts.best_model_path)
    LOGGER.info("Saved final model to %s", artifacts.final_model_path)
    LOGGER.info("Saved training log to %s", artifacts.training_log_path)
    LOGGER.info("Saved training metadata to %s", artifacts.training_metadata_path)
    LOGGER.info("Epochs completed: %d", artifacts.epochs_completed)


if __name__ == "__main__":
    main()
