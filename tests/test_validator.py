from src.data.partitioner import generate_partitions
from src.data.preprocessor import preprocess_dataset
from src.data.validator import validate_processed_data


def test_validator_passes_on_generated_phase1_artifacts(phase1_fixture: dict) -> None:
    preprocess_dataset(phase1_fixture["model_config"], phase1_fixture["paths_config"])
    generate_partitions(phase1_fixture["fl_config"], phase1_fixture["paths_config"])

    results = validate_processed_data(phase1_fixture["paths_config"])
    assert all(result.passed for result in results)
