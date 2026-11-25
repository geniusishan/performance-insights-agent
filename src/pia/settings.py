"""Configuration: YAML policy + semantic layer, environment for secrets and paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parent
CONFIG = PKG / "config"
# repo root when run from a clone; current folder when installed as a package
ROOT = PKG.parents[1] if (PKG.parents[1] / "pyproject.toml").exists() else Path.cwd()


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


@lru_cache(maxsize=1)
def policy() -> dict:
    return yaml.safe_load((CONFIG / "policy.yaml").read_text())


@lru_cache(maxsize=1)
def semantic_layer() -> dict:
    return yaml.safe_load((CONFIG / "semantic_layer.yaml").read_text())


def warehouse_path() -> Path:
    """Explicit PIA_WAREHOUSE > sibling Field Performance Hub clone > local data/ (cloud deploys)."""
    if os.environ.get("PIA_WAREHOUSE"):
        return Path(os.environ["PIA_WAREHOUSE"])
    sibling = ROOT.parent / "field-performance-hub" / "data" / "warehouse.duckdb"
    return sibling if sibling.exists() else ROOT / "data" / "warehouse.duckdb"


def audit_path() -> Path:
    p = Path(os.environ.get("PIA_AUDIT_DIR", ROOT / "audit"))
    p.mkdir(parents=True, exist_ok=True)
    return p
