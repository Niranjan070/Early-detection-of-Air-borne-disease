from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}
    config["__config_path__"] = str(path)
    return config


def resolve_path(config: dict[str, Any], raw_path: str | Path | None) -> Path | None:
    if raw_path in (None, ""):
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    config_dir = Path(config["__config_path__"]).parent
    return (config_dir / candidate).resolve()
