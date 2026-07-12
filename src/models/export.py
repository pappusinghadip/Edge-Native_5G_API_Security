"""TensorFlow Lite export and latency benchmarking for the centralized model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from src.evaluation.latency import benchmark_tflite_model
from src.models.io import load_processed_split, resolve_model_artifact_paths
from src.utils.config import load_yaml
from src.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportArtifacts:
    tflite_model_path: Path
    latency_metrics_path: Path


def export_tflite_model(model_config: dict[str, Any], paths_config: dict[str, Any]) -> ExportArtifacts:
    """Convert the centralized model to TFLite and benchmark local inference latency."""
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TensorFlow is required for TFLite export.") from exc

    paths = resolve_model_artifact_paths(paths_config, model_config)
    if not paths.best_model_path.exists():
        raise FileNotFoundError(f"Centralized model not found at {paths.best_model_path}")

    from src.models.cnn import focal_loss

    model = tf.keras.models.load_model(
        paths.best_model_path,
        custom_objects={"focal_loss": focal_loss()},
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    paths.tflite_model_path.write_bytes(tflite_model)

    X_test, _, _ = load_processed_split(paths.test_split_path)
    sample_pool = X_test[: min(len(X_test), 2048)]
    export_cfg = model_config["export"]
    latency_metrics = benchmark_tflite_model(
        paths.tflite_model_path,
        sample_pool,
        warmup_runs=int(export_cfg["warmup_runs"]),
        benchmark_runs=int(export_cfg["benchmark_runs"]),
    )
    paths.latency_metrics_path.write_text(json.dumps(latency_metrics, indent=2), encoding="utf-8")

    return ExportArtifacts(
        tflite_model_path=paths.tflite_model_path,
        latency_metrics_path=paths.latency_metrics_path,
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
    artifacts = export_tflite_model(model_config, paths_config)
    LOGGER.info("Saved TFLite model to %s", artifacts.tflite_model_path)
    LOGGER.info("Saved latency metrics to %s", artifacts.latency_metrics_path)


if __name__ == "__main__":
    main()
