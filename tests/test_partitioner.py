import numpy as np

from src.data.partitioner import generate_partitions, partition_dirichlet_indices, partition_iid_indices
from src.data.preprocessor import preprocess_dataset


def test_iid_partitions_cover_all_indices_once() -> None:
    partitions = partition_iid_indices(num_samples=100, num_clients=5, seed=42)
    merged = np.concatenate(partitions)
    assert len(merged) == 100
    assert len(np.unique(merged)) == 100


def test_dirichlet_partitions_cover_all_indices_once() -> None:
    labels = np.array([0] * 50 + [1] * 50)
    partitions = partition_dirichlet_indices(labels, num_clients=5, alpha=0.5, seed=42)
    merged = np.concatenate(partitions)
    assert len(merged) == len(labels)
    assert len(np.unique(merged)) == len(labels)


def test_generate_partitions_saves_iid_and_dirichlet_outputs(phase1_fixture: dict) -> None:
    preprocess_dataset(phase1_fixture["model_config"], phase1_fixture["paths_config"])
    manifest = generate_partitions(phase1_fixture["fl_config"], phase1_fixture["paths_config"])

    assert len(manifest["iid_files"]) == 5
    assert len(manifest["dirichlet_files"]) == 5
    assert (phase1_fixture["partitions_dir"] / "partition_manifest.json").exists()
    assert (phase1_fixture["figures_dir"] / "partition_distribution_iid.png").exists()
