"""Configuration helpers for YAML-driven project modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def resolve_project_root() -> Path:
    """Return the repository root based on this file location."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path_str: str | Path) -> Path:
    """Resolve a repository-relative path against the project root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return resolve_project_root() / path


def load_yaml(path_str: str | Path) -> dict[str, Any]:
    """Load a YAML document into a dictionary."""
    path = resolve_path(path_str)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
