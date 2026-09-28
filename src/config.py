"""Загрузка конфигурации и работа с путями проекта."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Загрузить YAML и проверить обязательные параметры."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    required = {"random_seed", "validation_size", "batch_size", "final_top_k", "fusion"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    if not 0 < float(config["validation_size"]) < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if not 1 <= int(config["final_top_k"]) <= 50:
        raise ValueError("final_top_k must be in [1, 50]")
    return deepcopy(config)


def resolve_project_path(value: str | Path) -> Path:
    """Преобразовать относительный путь от корня проекта."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
