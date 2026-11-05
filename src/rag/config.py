"""Settings loading with env-var overrides (RAG__SECTION__KEY=value)."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _find_root() -> Path:
    env = os.environ.get("RAG_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "settings.yaml").exists():
            return parent
    return Path.cwd()


ROOT = _find_root()


def resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path (absolute paths are returned unchanged)."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _apply_env_overrides(settings: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(settings)
    for key, raw in os.environ.items():
        if not key.startswith("RAG__"):
            continue
        parts = [p.lower() for p in key[5:].split("__") if p]
        if not parts:
            continue
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                break
        else:
            node[parts[-1]] = _coerce(raw)
    return out


@lru_cache(maxsize=8)
def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    p = resolve(path or os.environ.get("RAG_SETTINGS", "configs/settings.yaml"))
    return _apply_env_overrides(_load_yaml(str(p)))


def load_yaml(path: str | Path) -> dict[str, Any]:
    return copy.deepcopy(_load_yaml(str(resolve(path))))
