"""Config loading. YAML files live in config/, paths resolve from the repo root."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Repo root = three levels up from this file (src/vnresearch/config.py).
#
# VNRESEARCH_ROOT overrides it for callers that import this package from
# somewhere else. Without the override an outside importer — vnstock-paper —
# silently resolves `config/` and `data/` back into this repo, which happens to
# be what it wants for features and models but is not something to leave to
# coincidence. Unset, behaviour is exactly as before.
ROOT = Path(os.environ.get("VNRESEARCH_ROOT") or Path(__file__).resolve().parents[2])
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
