"""Flower client entrypoint scaffold."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

from src.utils.config import load_yaml
from src.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClientRuntimeConfig:
    client_id: str
    server_address: str


def runtime_config_from_env() -> ClientRuntimeConfig:
    return ClientRuntimeConfig(
        client_id=os.getenv("CLIENT_ID", "0"),
        server_address=os.getenv("SERVER_ADDRESS", "localhost:8080"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/fl.yaml")
    parser.add_argument("--paths", required=True, help="Path to configs/paths.yaml")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    _ = load_yaml(args.config)
    _ = load_yaml(args.paths)
    runtime = runtime_config_from_env()
    LOGGER.info("FL client scaffold initialized for client %s.", runtime.client_id)
    LOGGER.info("Server target: %s", runtime.server_address)
    LOGGER.info("Flower client execution is not implemented yet.")


if __name__ == "__main__":
    main()
