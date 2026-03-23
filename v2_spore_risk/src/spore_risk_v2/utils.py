from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def slugify_label(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def pick_first_present(columns: Iterable[str], frame: pd.DataFrame) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def choose_join_key(left: pd.DataFrame, right: pd.DataFrame, priority: Iterable[str]) -> str:
    for key in priority:
        if key in left.columns and key in right.columns:
            return key
    shared = [column for column in left.columns if column in right.columns]
    if not shared:
        raise ValueError("No shared key was found to join the tables.")
    return shared[0]
