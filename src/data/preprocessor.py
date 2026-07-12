"""Preprocessing pipeline for CICIoT2023 -> 1D-CNN training tensors."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.data.download import discover_csv_files
from src.utils.config import load_yaml, resolve_path
from src.utils.logger import configure_logging
from src.utils.seed import set_global_seed

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = (
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "ack_count",
    "syn_count",
    "Tot sum",
    "AVG",
    "IAT",
    "Number",
)

SPLIT_RATIOS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class PreprocessingPlan:
    features: Sequence[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    input_shape: tuple[int, int]
    label_column: str
    negative_label: str
    random_seed: int


@dataclass(frozen=True)
class SavedSplit:
    name: str
    path: Path
    size: int


@dataclass(frozen=True)
class PreprocessingArtifacts:
    train_path: Path
    val_path: Path
    test_path: Path
    scaler_path: Path
    metadata_path: Path
    dropped_rows: int
    total_rows: int
    infinite_rows: int


def build_preprocessing_plan(model_config: dict) -> PreprocessingPlan:
    dataset_cfg = model_config["dataset"]
    model_cfg = model_config["model"]
    return PreprocessingPlan(
        features=tuple(dataset_cfg["features"]),
        train_ratio=float(dataset_cfg["train_ratio"]),
        val_ratio=float(dataset_cfg["val_ratio"]),
        test_ratio=float(dataset_cfg["test_ratio"]),
        input_shape=tuple(model_cfg["input_shape"]),
        label_column=str(dataset_cfg.get("label_column", "label")),
        negative_label=str(dataset_cfg.get("negative_label", "Benign")),
        random_seed=int(model_config.get("random_seed", 42)),
    )


def infer_binary_label_from_path(path: Path, negative_label: str) -> int:
    """Infer a binary label from the filename when the CSV has no label column."""
    filename = path.stem.casefold()
    negative_token = negative_label.casefold()
    return 0 if negative_token in filename else 1


def load_raw_data(raw_dir: Path, features: Sequence[str], label_column: str, negative_label: str) -> pd.DataFrame:
    """Load and concatenate raw CSV files using configured features and label strategy."""
    csv_files = discover_csv_files(raw_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    frames: list[pd.DataFrame] = []
    feature_list = list(features)
    for path in csv_files:
        header = pd.read_csv(path, nrows=0)
        missing_features = [column for column in feature_list if column not in header.columns]
        if missing_features:
            raise ValueError(
                f"{path.name} is missing configured feature columns: {missing_features}. "
                f"Available columns: {header.columns.tolist()}"
            )

        if label_column in header.columns:
            frame = pd.read_csv(path, usecols=feature_list + [label_column], low_memory=False)
        else:
            frame = pd.read_csv(path, usecols=feature_list, low_memory=False)
            frame[label_column] = infer_binary_label_from_path(path, negative_label)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def encode_labels(labels: pd.Series, negative_label: str) -> np.ndarray:
    """Convert labels to binary values where benign is 0 and everything else is 1."""
    if pd.api.types.is_numeric_dtype(labels):
        unique_values = set(labels.dropna().astype(int).unique().tolist())
        if unique_values.issubset({0, 1}):
            return labels.astype(int).to_numpy()

    normalized = labels.astype(str).str.strip().str.casefold()
    benign = negative_label.strip().casefold()
    return (normalized != benign).astype(int).to_numpy()


def reshape_for_cnn(features: np.ndarray, input_shape: tuple[int, int]) -> np.ndarray:
    """Reshape flat feature rows into CNN input tensors."""
    time_steps, channels = input_shape
    expected_columns = time_steps * channels
    if features.shape[1] != expected_columns:
        raise ValueError(
            f"Cannot reshape feature matrix with {features.shape[1]} columns into input shape {input_shape}"
        )
    return features.reshape(features.shape[0], time_steps, channels)


def save_split(path: Path, X: np.ndarray, y: np.ndarray, indices: np.ndarray) -> SavedSplit:
    """Save a processed split as a compressed NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, indices=indices)
    return SavedSplit(name=path.stem, path=path, size=int(len(y)))


def preprocess_dataset(model_config: dict, paths_config: dict) -> PreprocessingArtifacts:
    """Run the end-to-end preprocessing pipeline and persist all artifacts."""
    plan = build_preprocessing_plan(model_config)
    set_global_seed(plan.random_seed)

    raw_dir = resolve_path(paths_config["data"]["raw"])
    processed_dir = resolve_path(paths_config["data"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_frame = load_raw_data(raw_dir, plan.features, plan.label_column, plan.negative_label)
    total_rows = len(raw_frame)
    feature_frame = raw_frame[list(plan.features)].apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    label_values = raw_frame[plan.label_column]
    infinite_rows = int((~np.isfinite(feature_frame.to_numpy(dtype=np.float64))).any(axis=1).sum())
    valid_mask = ~(feature_frame.isna().any(axis=1) | label_values.isna())

    cleaned_features = feature_frame.loc[valid_mask].to_numpy(dtype=np.float64)
    cleaned_labels = encode_labels(label_values.loc[valid_mask], plan.negative_label).astype(np.int64)
    cleaned_indices = raw_frame.index.to_numpy(dtype=np.int64)[valid_mask.to_numpy()]
    dropped_rows = int(total_rows - len(cleaned_features))

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        cleaned_features,
        cleaned_labels,
        cleaned_indices,
        test_size=plan.val_ratio + plan.test_ratio,
        stratify=cleaned_labels,
        random_state=plan.random_seed,
    )

    val_share_of_temp = plan.val_ratio / (plan.val_ratio + plan.test_ratio)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp,
        y_temp,
        idx_temp,
        test_size=1.0 - val_share_of_temp,
        stratify=y_temp,
        random_state=plan.random_seed,
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    X_train_ready = reshape_for_cnn(X_train_scaled, plan.input_shape)
    X_val_ready = reshape_for_cnn(X_val_scaled, plan.input_shape)
    X_test_ready = reshape_for_cnn(X_test_scaled, plan.input_shape)

    train_path = processed_dir / "train.npz"
    val_path = processed_dir / "val.npz"
    test_path = processed_dir / "test.npz"
    scaler_path = processed_dir / "scaler.pkl"
    metadata_path = processed_dir / "preprocessing_metadata.json"

    saved_splits = (
        save_split(train_path, X_train_ready, y_train, idx_train),
        save_split(val_path, X_val_ready, y_val, idx_val),
        save_split(test_path, X_test_ready, y_test, idx_test),
    )

    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)

    metadata: dict[str, Any] = {
        "features": list(plan.features),
        "input_shape": list(plan.input_shape),
        "total_rows": total_rows,
        "dropped_rows": dropped_rows,
        "infinite_rows": infinite_rows,
        "split_sizes": {split.name: split.size for split in saved_splits},
        "label_column": plan.label_column,
        "negative_label": plan.negative_label,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return PreprocessingArtifacts(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
        dropped_rows=dropped_rows,
        total_rows=total_rows,
        infinite_rows=infinite_rows,
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
    artifacts = preprocess_dataset(model_config, paths_config)
    LOGGER.info("Saved train split to %s", artifacts.train_path)
    LOGGER.info("Saved val split to %s", artifacts.val_path)
    LOGGER.info("Saved test split to %s", artifacts.test_path)
    LOGGER.info("Dropped %d/%d rows with missing data.", artifacts.dropped_rows, artifacts.total_rows)
    LOGGER.info("Rows containing infinite feature values: %d", artifacts.infinite_rows)


if __name__ == "__main__":
    main()
