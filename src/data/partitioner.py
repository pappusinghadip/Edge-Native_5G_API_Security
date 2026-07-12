"""Data partitioning helpers for IID and Dirichlet FL simulations."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import load_yaml, resolve_path
from src.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)
PARTITION_FILE_RE = re.compile(r"client_(?P<client>\d+)_(?P<mode>iid|dir_[A-Za-z0-9p]+)\.npz$")


def partition_iid_indices(num_samples: int, num_clients: int, seed: int = 42) -> list[np.ndarray]:
    """Return shuffled IID index partitions."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    return [chunk.copy() for chunk in np.array_split(indices, num_clients)]


def partition_dirichlet_indices(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[np.ndarray]:
    """Return non-IID partitions using class-wise Dirichlet allocation."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    labels = np.asarray(labels)
    if len(labels) < num_clients:
        raise ValueError("Need at least one sample per client for Dirichlet partitioning")

    for attempt in range(50):
        rng = np.random.default_rng(seed + attempt)
        client_bins: list[list[int]] = [[] for _ in range(num_clients)]

        for class_id in np.unique(labels):
            class_indices = np.flatnonzero(labels == class_id)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            raw_counts = proportions * len(class_indices)
            counts = np.floor(raw_counts).astype(int)

            deficit = len(class_indices) - int(counts.sum())
            if deficit > 0:
                order = np.argsort(raw_counts - counts)[::-1]
                for client_id in order[:deficit]:
                    counts[client_id] += 1

            start = 0
            for client_id, count in enumerate(counts):
                stop = start + int(count)
                client_bins[client_id].extend(class_indices[start:stop].tolist())
                start = stop

        if all(len(bin_indices) > 0 for bin_indices in client_bins):
            partitions: list[np.ndarray] = []
            for client_indices in client_bins:
                client_array = np.asarray(client_indices, dtype=int)
                rng.shuffle(client_array)
                partitions.append(client_array)
            return partitions

    raise RuntimeError("Unable to generate non-empty Dirichlet partitions after 50 attempts")


def load_processed_split(split_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X, y, and indices from a processed split archive."""
    with np.load(split_path) as data:
        return data["X"], data["y"], data["indices"]


def alpha_tag(alpha: float) -> str:
    """Create a filesystem-safe alpha tag."""
    return str(alpha).replace(".", "p")


def save_partition_files(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    partitions: list[np.ndarray],
    output_dir: Path,
    mode: str,
) -> list[Path]:
    """Persist client partitions as compressed NumPy archives."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for client_id, partition_indices in enumerate(partitions):
        file_path = output_dir / f"client_{client_id}_{mode}.npz"
        np.savez_compressed(
            file_path,
            X=X[partition_indices],
            y=y[partition_indices],
            indices=indices[partition_indices],
        )
        saved_paths.append(file_path)
    return saved_paths


def summarize_partition_labels(y: np.ndarray, partitions: list[np.ndarray]) -> list[dict[str, int]]:
    """Return benign/malicious counts for each client partition."""
    summary: list[dict[str, int]] = []
    for client_indices in partitions:
        labels = y[client_indices]
        benign = int(np.sum(labels == 0))
        malicious = int(np.sum(labels == 1))
        summary.append({"benign": benign, "malicious": malicious})
    return summary


def plot_partition_distribution(summary: list[dict[str, int]], figure_path: Path, title: str) -> None:
    """Plot benign vs malicious counts per client."""
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    client_labels = [f"client_{index}" for index in range(len(summary))]
    benign_counts = [item["benign"] for item in summary]
    malicious_counts = [item["malicious"] for item in summary]

    x = np.arange(len(summary))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, benign_counts, width=width, label="Benign")
    ax.bar(x + width / 2, malicious_counts, width=width, label="Malicious")
    ax.set_xticks(x, client_labels)
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def generate_partitions(fl_config: dict, paths_config: dict) -> dict[str, Any]:
    """Generate and save IID and Dirichlet partitions from the training split."""
    federation_cfg = fl_config["federation"]
    partition_cfg = fl_config["partitioning"]

    train_path = resolve_path(paths_config["data"]["processed"]) / "train.npz"
    output_dir = resolve_path(paths_config["data"]["partitions"])
    figures_dir = resolve_path(paths_config["results"]["figures"])

    X_train, y_train, train_indices = load_processed_split(train_path)
    num_clients = int(federation_cfg["num_clients"])
    seed = int(fl_config["random_seed"])
    alpha = float(partition_cfg["dirichlet_alpha"])

    iid_partitions = partition_iid_indices(len(y_train), num_clients=num_clients, seed=seed)
    dir_partitions = partition_dirichlet_indices(y_train, num_clients=num_clients, alpha=alpha, seed=seed)

    iid_paths = save_partition_files(X_train, y_train, train_indices, iid_partitions, output_dir, mode="iid")
    dir_mode = f"dir_{alpha_tag(alpha)}"
    dir_paths = save_partition_files(X_train, y_train, train_indices, dir_partitions, output_dir, mode=dir_mode)

    iid_summary = summarize_partition_labels(y_train, iid_partitions)
    dir_summary = summarize_partition_labels(y_train, dir_partitions)
    plot_partition_distribution(iid_summary, figures_dir / "partition_distribution_iid.png", "IID Client Distribution")
    plot_partition_distribution(
        dir_summary,
        figures_dir / f"partition_distribution_{dir_mode}.png",
        f"Dirichlet Client Distribution (alpha={alpha})",
    )

    manifest = {
        "train_path": str(train_path),
        "num_clients": num_clients,
        "iid_files": [str(path) for path in iid_paths],
        "dirichlet_files": [str(path) for path in dir_paths],
        "dirichlet_alpha": alpha,
        "iid_summary": iid_summary,
        "dirichlet_summary": dir_summary,
    }
    manifest_path = output_dir / "partition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/fl.yaml")
    parser.add_argument("--paths", required=True, help="Path to configs/paths.yaml")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    fl_config = load_yaml(args.config)
    paths_config = load_yaml(args.paths)
    manifest = generate_partitions(fl_config, paths_config)
    LOGGER.info("Saved IID partitions: %d", len(manifest["iid_files"]))
    LOGGER.info("Saved Dirichlet partitions: %d", len(manifest["dirichlet_files"]))


if __name__ == "__main__":
    main()
