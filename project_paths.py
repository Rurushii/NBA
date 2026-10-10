"""Portable filesystem paths used by the NBA prediction pipeline.

By default, source data and generated artifacts live beside these scripts so a
fresh clone can run without machine-specific absolute paths.

Environment variables can override those locations when needed:
  - NBA_DATA_DIR: directory containing source inputs such as nba_games_raw.csv
  - NBA_OUTPUT_DIR: directory for generated features, models, JSON, and HTML
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _directory_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    directory = Path(value).expanduser() if value else default
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    return directory.resolve()


DATA_DIR = _directory_from_env("NBA_DATA_DIR", PROJECT_ROOT)
OUTPUT_DIR = _directory_from_env("NBA_OUTPUT_DIR", PROJECT_ROOT)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def data_path(filename: str) -> Path:
    """Return a path to a source-data file."""
    return DATA_DIR / filename


def output_path(filename: str) -> Path:
    """Return a path to a generated artifact."""
    return OUTPUT_DIR / filename
