import json

import numpy as np

from src.data.preprocessor import (
    FEATURE_COLUMNS,
    SPLIT_RATIOS,
    build_preprocessing_plan,
    preprocess_dataset,
)


def test_feature_column_count() -> None:
    assert len(FEATURE_COLUMNS) == 10


def test_split_ratios_sum_to_one() -> None:
    assert round(sum(SPLIT_RATIOS), 2) == 1.00


def test_preprocessing_plan_matches_config_shape() -> None:
    config = {
        "dataset": {
            "features": list(FEATURE_COLUMNS),
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "model": {"input_shape": [10, 1]},
    }
    plan = build_preprocessing_plan(config)
    assert plan.input_shape == (10, 1)


def test_preprocess_dataset_saves_expected_artifacts(phase1_fixture: dict) -> None:
    artifacts = preprocess_dataset(phase1_fixture["model_config"], phase1_fixture["paths_config"])

    assert artifacts.train_path.exists()
    assert artifacts.val_path.exists()
    assert artifacts.test_path.exists()
    assert artifacts.scaler_path.exists()
    assert artifacts.metadata_path.exists()

    with np.load(artifacts.train_path) as train:
        assert train["X"].shape[1:] == (10, 1)
        assert set(train["y"].tolist()) == {0, 1}

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["dropped_rows"] == 1
