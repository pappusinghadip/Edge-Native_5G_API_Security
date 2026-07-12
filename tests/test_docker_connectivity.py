from pathlib import Path


def test_docker_scaffold_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "docker" / "compose.yaml").exists()
    assert (root / "docker" / "Dockerfile.server").exists()
    assert (root / "docker" / "Dockerfile.client").exists()
