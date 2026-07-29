"""Config loading. YAML files live in config/, paths resolve from the repo root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# repo root = three levels up from this file (src/vnresearch/config.py)
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


def load(name: str) -> dict[str, Any]:
    """Load config/<name>.yaml."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dsn(cfg: dict[str, Any] | None = None) -> str:
    """Postgres DSN. The environment wins so a run can point at another database."""
    if env := os.environ.get("DATABASE_URL"):
        return env
    cfg = cfg if cfg is not None else load("data")
    return cfg["dsn"]


def path(rel: str) -> Path:
    """Resolve a repo-relative path from config."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
