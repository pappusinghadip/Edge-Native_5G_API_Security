from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import FEATURE_COLUMNS


@pytest.fixture()
def phase1_fixture(tmp_path: Path) -> dict:
    data_root = tmp_path / "data"
    raw_dir = data_root / "raw"
    processed_dir = data_root / "processed"
    partitions_dir = data_root / "partitions"
    results_root = tmp_path / "results"
    figures_dir = results_root / "figures"

    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    partitions_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    (results_root / "metrics").mkdir(parents=True)
    (results_root / "models").mkdir(parents=True)
    (results_root / "tensorboard").mkdir(parents=True)

    rows: list[dict[str, float | str]] = []
    for index in range(120):
        label = "Benign" if index < 60 else ("BruteForce" if index < 90 else "DoS")
        offset = 0.0 if label == "Benign" else (20.0 if label == "BruteForce" else 35.0)
        row = {
            "Header_Length": offset + index,
            "Protocol Type": offset + 1 + (index % 7),
            "Time_To_Live": offset + 2 + (index % 5),
            "Rate": offset + 3 + (index % 11),
            "ack_count": offset + 4 + (index % 13),
            "syn_count": offset + 5 + (index % 17),
            "Tot sum": offset + 6 + (index % 19),
            "AVG": offset + 7 + (index % 23),
            "IAT": offset + 8 + (index % 29),
            "Number": offset + 9 + (index % 31),
            "label": label,
        }
        rows.append(row)

    rows[5]["Header_Length"] = np.nan
    frame = pd.DataFrame(rows)
    frame.iloc[:60].to_csv(raw_dir / "part_1.csv", index=False)
    frame.iloc[60:].to_csv(raw_dir / "part_2.csv", index=False)

    model_config = {
        "random_seed": 42,
        "dataset": {
            "label_column": "label",
            "negative_label": "Benign",
            "features": list(FEATURE_COLUMNS),
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "model": {"input_shape": [10, 1]},
    }
    fl_config = {
        "random_seed": 42,
        "federation": {"num_clients": 5},
        "partitioning": {"dirichlet_alpha": 0.5},
    }
    paths_config = {
        "data": {
            "raw": str(raw_dir),
            "processed": str(processed_dir),
            "partitions": str(partitions_dir),
        },
        "results": {
            "figures": str(figures_dir),
            "metrics": str(results_root / "metrics"),
            "models": str(results_root / "models"),
            "tensorboard": str(results_root / "tensorboard"),
        },
    }

    return {
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "partitions_dir": partitions_dir,
        "figures_dir": figures_dir,
        "model_config": model_config,
        "fl_config": fl_config,
        "paths_config": paths_config,
    }
