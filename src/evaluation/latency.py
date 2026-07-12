"""Latency benchmarking helpers for TensorFlow and TensorFlow Lite models."""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import load_yaml
from src.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


def summarize_latency(latencies_ms: list[float]) -> dict[str, float]:
    """Summarize latency samples in milliseconds."""
    if not latencies_ms:
        raise ValueError("No latency samples were recorded")

    array = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(statistics.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def benchmark_tflite_model(
    model_path: Path,
    samples: np.ndarray,
    warmup_runs: int,
    benchmark_runs: int,
) -> dict[str, Any]:
    """Benchmark a TFLite model with single-sample inference runs."""
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("TensorFlow is required for TFLite benchmarking.") from exc

    if samples.ndim < 3:
        raise ValueError("Expected samples shaped like (N, 1, 10)")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_dtype = input_details["dtype"]

    sample_count = len(samples)
    latencies_ms: list[float] = []
    total_runs = warmup_runs + benchmark_runs

    for run_index in range(total_runs):
        sample = samples[run_index % sample_count : (run_index % sample_count) + 1].astype(input_dtype)
        start = time.perf_counter_ns()
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        if run_index >= warmup_runs:
            latencies_ms.append(elapsed_ms)

    summary = summarize_latency(latencies_ms)
    summary.update(
        {
            "warmup_runs": int(warmup_runs),
            "benchmark_runs": int(benchmark_runs),
            "samples_used": int(sample_count),
        }
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/docker.yaml")
    parser.add_argument("--paths", required=True, help="Path to configs/paths.yaml")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    _ = load_yaml(args.config)
    _ = load_yaml(args.paths)
    LOGGER.info("Latency helpers are available for TFLite export and Docker benchmarking.")


if __name__ == "__main__":
    main()
