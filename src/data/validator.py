"""Data validation for processed splits and FL partitions."""

from __future__ import annotations

import argparse
import pickle
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

from src.data.partitioner import PARTITION_FILE_RE
from src.utils.config import load_yaml
from src.utils.logger import configure_logging
from src.utils.config import resolve_path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    expected_result: str
    failure_reason: str


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    details: str


REQUIRED_CHECKS = (
    ValidationCheck("no_nan", "No NaN values in any split", "Preprocessing bug"),
    ValidationCheck("input_shape", "X shape ends with (10, 1)", "Reshape error"),
    ValidationCheck("train_range", "X_train values stay within [0, 1]", "Scaler misfit"),
    ValidationCheck("binary_labels", "y contains only 0 and 1", "Label encoding error"),
    ValidationCheck("split_overlap", "Train/val/test indices do not overlap", "Data leakage"),
    ValidationCheck("ratio_preserved", "Class ratio preserved across splits", "Stratification failed"),
    ValidationCheck("partition_sizes", "Partitions sum to train size", "Partition logic error"),
    ValidationCheck("scaler_saved", "Scaler artifact loads successfully", "Serialization error"),
)


def load_split(split_path: Path) -> dict[str, np.ndarray]:
    """Load a processed split archive into memory."""
    with np.load(split_path) as data:
        return {key: data[key] for key in data.files}


def group_partition_files(partitions_dir: Path) -> dict[str, list[Path]]:
    """Group partition files by mode suffix."""
    groups: dict[str, list[Path]] = {}
    for path in sorted(partitions_dir.glob("client_*.npz")):
        match = PARTITION_FILE_RE.match(path.name)
        if not match:
            continue
        groups.setdefault(match.group("mode"), []).append(path)
    return groups


def validate_processed_data(paths_config: dict) -> list[ValidationResult]:
    """Run the eight required validation checks."""
    processed_dir = resolve_path(paths_config["data"]["processed"])
    partitions_dir = resolve_path(paths_config["data"]["partitions"])

    train = load_split(processed_dir / "train.npz")
    val = load_split(processed_dir / "val.npz")
    test = load_split(processed_dir / "test.npz")
    splits = {"train": train, "val": val, "test": test}
    scaler_path = processed_dir / "scaler.pkl"

    results: list[ValidationResult] = []

    no_nan_passed = all(not np.isnan(split["X"]).any() for split in splits.values())
    results.append(ValidationResult("no_nan", no_nan_passed, "Checked all split feature tensors for NaN values."))

    expected_shape = (10, 1)
    shape_passed = all(tuple(split["X"].shape[1:]) == expected_shape for split in splits.values())
    results.append(ValidationResult("input_shape", shape_passed, f"Expected split shape suffix {expected_shape}."))

    train_min = float(np.min(train["X"]))
    train_max = float(np.max(train["X"]))
    range_passed = train_min >= -1e-6 and train_max <= 1.0 + 1e-6
    results.append(ValidationResult("train_range", range_passed, f"Observed train range [{train_min:.8f}, {train_max:.8f}]."))

    unique_labels = sorted({int(value) for split in splits.values() for value in split["y"]})
    labels_passed = unique_labels == [0, 1]
    results.append(ValidationResult("binary_labels", labels_passed, f"Observed label set {unique_labels}."))

    train_idx = set(train["indices"].tolist())
    val_idx = set(val["indices"].tolist())
    test_idx = set(test["indices"].tolist())
    overlap_passed = not train_idx.intersection(val_idx) and not train_idx.intersection(test_idx) and not val_idx.intersection(test_idx)
    results.append(ValidationResult("split_overlap", overlap_passed, "Train/val/test indices are disjoint."))

    contingency = np.array(
        [
            [int(np.sum(split["y"] == 0)), int(np.sum(split["y"] == 1))]
            for split in (train, val, test)
        ]
    )
    _, p_value, _, _ = chi2_contingency(contingency)
    ratio_passed = bool(p_value > 0.05)
    results.append(ValidationResult("ratio_preserved", ratio_passed, f"Chi-squared p-value={p_value:.4f}."))

    partition_groups = group_partition_files(partitions_dir)
    partition_sizes_passed = bool(partition_groups)
    partition_details: list[str] = []
    train_size = len(train["indices"])
    for mode, files in partition_groups.items():
        grouped_indices: list[np.ndarray] = []
        total = 0
        for file_path in files:
            with np.load(file_path) as partition_data:
                indices = partition_data["indices"]
                grouped_indices.append(indices)
                total += int(len(indices))
        merged = np.concatenate(grouped_indices) if grouped_indices else np.array([], dtype=int)
        mode_ok = total == train_size and len(np.unique(merged)) == train_size
        partition_sizes_passed = partition_sizes_passed and mode_ok
        partition_details.append(f"{mode}: total={total}, unique={len(np.unique(merged))}")
    results.append(ValidationResult("partition_sizes", partition_sizes_passed, "; ".join(partition_details) or "No partition files found."))

    scaler_saved_passed = False
    scaler_details = f"Missing scaler at {scaler_path}"
    if scaler_path.exists():
        with scaler_path.open("rb") as handle:
            scaler = pickle.load(handle)
        scaler_saved_passed = hasattr(scaler, "transform")
        scaler_details = f"Loaded scaler type {type(scaler).__name__}."
    results.append(ValidationResult("scaler_saved", scaler_saved_passed, scaler_details))

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/paths.yaml")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    paths_config = load_yaml(args.config)
    results = validate_processed_data(paths_config)
    failures = [result for result in results if not result.passed]

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        LOGGER.info("[%s] %s :: %s", status, result.name, result.details)

    if failures:
        raise SystemExit(f"Validation failed for {len(failures)} checks.")


if __name__ == "__main__":
    main()
