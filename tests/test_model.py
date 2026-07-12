import copy
import json

import numpy as np
import pytest

from src.data.preprocessor import preprocess_dataset
from src.models.cnn import MODEL_SPEC, build_model, model_summary_to_dict
from src.models.evaluate import evaluate_centralized_model
from src.models.export import export_tflite_model
from src.models.train import oversample_minority_class, train_centralized_model


def test_model_spec_matches_thesis_design() -> None:
    assert MODEL_SPEC["input_shape"] == (10, 1)
    assert MODEL_SPEC["num_classes"] == 2
    assert MODEL_SPEC["conv_filters"] == (64, 128, 128)


def test_model_builds_with_expected_output_shape() -> None:
    tf = pytest.importorskip("tensorflow")
    model = build_model()
    assert model.output_shape == (None, 2)
    assert model.count_params() < 300000
    summary = model_summary_to_dict(model)
    assert summary["total_params"] == model.count_params()
    assert summary["layers"]


def test_oversample_minority_class_balances_binary_labels() -> None:
    X = np.arange(12, dtype=np.float32).reshape(6, 1, 2)
    y = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    X_resampled, y_resampled, counts = oversample_minority_class(X, y, target_ratio=1.0, seed=42)
    assert X_resampled.shape[0] == 8
    assert counts == {"0": 4, "1": 4}
    assert int((y_resampled == 0).sum()) == 4
    assert int((y_resampled == 1).sum()) == 4


def test_phase2_smoke_pipeline(phase1_fixture: dict) -> None:
    pytest.importorskip("tensorflow")

    preprocess_dataset(phase1_fixture["model_config"], phase1_fixture["paths_config"])

    model_config = copy.deepcopy(phase1_fixture["model_config"])
    model_config["model"].update(
        {
            "num_classes": 2,
            "conv_filters": [64, 128, 128],
            "kernel_size": 3,
            "dense_units": [256, 128],
            "dropout_rates": [0.4, 0.3],
            "optimizer": "adam",
            "learning_rate": 0.001,
            "momentum": 0.9,
            "use_batch_norm": True,
            "use_focal_loss": True,
            "focal_alpha": 0.75,
            "focal_gamma": 2.0,
            "label_smoothing": 0.1,
        }
    )
    model_config["training"] = {
        "epochs": 1,
        "batch_size": 16,
        "class_weighting": "none",
        "resampling_strategy": "oversample_minority",
        "oversampling_target_ratio": 1.0,
        "tensorboard_subdir": "test-centralized",
        "checkpoint_name": "centralized_best_test.h5",
        "final_model_name": "centralized_final_test.h5",
        "training_log_name": "centralized_training_log_test.csv",
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 1,
            "restore_best_weights": True,
        },
    }
    model_config["evaluation"] = {
        "metrics_name": "centralized_results_test.json",
        "summary_name": "model_summary_test.txt",
        "confusion_matrix_figure": "confusion_matrix_test.png",
        "roc_figure": "roc_test.png",
        "training_curves_figure": "training_curves_test.png",
    }
    model_config["export"] = {
        "tflite_name": "centralized_best_test.tflite",
        "latency_metrics_name": "latency_centralized_test.json",
        "warmup_runs": 2,
        "benchmark_runs": 5,
    }

    training_artifacts = train_centralized_model(model_config, phase1_fixture["paths_config"])
    assert training_artifacts.best_model_path.exists()
    assert training_artifacts.training_log_path.exists()
    assert training_artifacts.training_metadata_path.exists()

    evaluation_artifacts = evaluate_centralized_model(model_config, phase1_fixture["paths_config"])
    assert evaluation_artifacts.metrics_path.exists()
    assert evaluation_artifacts.summary_path.exists()
    assert evaluation_artifacts.confusion_matrix_figure_path.exists()
    assert evaluation_artifacts.roc_figure_path.exists()
    assert evaluation_artifacts.training_curves_figure_path.exists()

    metrics = json.loads(evaluation_artifacts.metrics_path.read_text(encoding="utf-8"))
    assert "f1_macro" in metrics
    assert "auc_roc" in metrics
    assert "decision_threshold" in metrics

    export_artifacts = export_tflite_model(model_config, phase1_fixture["paths_config"])
    assert export_artifacts.tflite_model_path.exists()
    latency_metrics = json.loads(export_artifacts.latency_metrics_path.read_text(encoding="utf-8"))
    assert latency_metrics["benchmark_runs"] == 5
