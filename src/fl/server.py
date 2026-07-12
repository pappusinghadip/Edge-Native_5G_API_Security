"""Flower server entrypoint scaffold."""

from __future__ import annotations

import argparse
import logging

from src.utils.config import load_yaml
from src.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/fl.yaml")
    parser.add_argument("--paths", required=True, help="Path to configs/paths.yaml")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    fl_config = load_yaml(args.config)
    _ = load_yaml(args.paths)
    rounds = int(fl_config["federation"]["num_rounds"])
    LOGGER.info("FL server scaffold initialized for %d rounds.", rounds)
    LOGGER.info("Flower server startup is not implemented yet.")


if __name__ == "__main__":
    main()
