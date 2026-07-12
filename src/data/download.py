"""Dataset acquisition and local staging helpers for CICIoT2023."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from src.utils.config import load_yaml, resolve_path
from src.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/paths.yaml")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Optional local directory containing CSV files to copy into data/raw.",
    )
    return parser


def ensure_raw_directory(paths_config: dict) -> Path:
    raw_dir = resolve_path(paths_config["data"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def discover_csv_files(directory: Path) -> list[Path]:
    """Return CSV files under a directory tree in deterministic order."""
    return sorted(path for path in directory.rglob("*.csv") if path.is_file())


def stage_local_dataset(source_dir: Path, raw_dir: Path) -> list[Path]:
    """Copy CSV files from a local source directory into the raw data directory."""
    source_files = discover_csv_files(source_dir)
    if not source_files:
        raise FileNotFoundError(f"No CSV files found in source directory: {source_dir}")

    copied: list[Path] = []
    for source_file in source_files:
        destination = raw_dir / source_file.name
        shutil.copy2(source_file, destination)
        copied.append(destination)
    return copied


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    paths_config = load_yaml(args.config)
    raw_dir = ensure_raw_directory(paths_config)
    LOGGER.info("Raw data directory ready at %s", raw_dir)

    if args.source_dir:
        copied = stage_local_dataset(resolve_path(args.source_dir), raw_dir)
        LOGGER.info("Copied %d CSV files into %s", len(copied), raw_dir)
        return

    csv_files = discover_csv_files(raw_dir)
    LOGGER.info("Found %d staged CSV files in %s", len(csv_files), raw_dir)
    LOGGER.info("Network download is intentionally left out; use --source-dir to stage data locally.")


if __name__ == "__main__":
    main()
