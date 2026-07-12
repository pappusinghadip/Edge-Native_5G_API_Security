"""Thesis figure generation scaffold."""

from __future__ import annotations

import argparse
import logging

from src.utils.config import load_yaml
from src.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/paths.yaml")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    _ = load_yaml(args.config)
    LOGGER.info("Plot generation scaffold initialized.")
    LOGGER.info("Publication-quality figure generation is not implemented yet.")


if __name__ == "__main__":
    main()
