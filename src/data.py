"""Загрузка parquet-файлов и проверка их схемы."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


FILES = {
    "train": "train.parquet",
    "queries": "benchmark_queries.parquet",
    "items": "benchmark_items.parquet",
}

QUERY_COLUMNS = {
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infn_params_text",
    "search_category",
}
ITEM_FEATURE_COLUMNS = {
    "item_title_raw",
    "item_description_raw",
    "item_infn_params_text",
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
}

# В исходных parquet встречается вариант infm вместо infn.
COLUMN_ALIASES = {
    "search_infm_params_text": "search_infn_params_text",
    "item_infm_params_text": "item_infn_params_text",
}


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    queries: pd.DataFrame
    items: pd.DataFrame


def required_paths(data_dir: str | Path) -> dict[str, Path]:
    root = Path(data_dir)
    return {name: root / filename for name, filename in FILES.items()}


def missing_data_files(data_dir: str | Path) -> list[Path]:
    return [path for path in required_paths(data_dir).values() if not path.is_file()]


def format_missing_data_message(data_dir: str | Path) -> str:
    root = Path(data_dir)
    names = ", ".join(FILES.values())
    return f"Data are not installed. Put these three files into {root}: {names}"


def _validate_columns(frame: pd.DataFrame, expected: Iterable[str], name: str) -> None:
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing expected columns: {missing}")


def _force_id_strings(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("query_id", "item_id"):
        if column in result:
            if result[column].isna().any():
                raise ValueError(f"{column} contains missing values")
            result[column] = result[column].astype("string")
    return result


def _canonicalize_columns(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for source, target in COLUMN_ALIASES.items():
        if source in frame.columns:
            if target in frame.columns:
                raise ValueError(
                    f"{name} contains both alias columns {source!r} and {target!r}; "
                    "cannot choose safely"
                )
            rename[source] = target
    if rename:
        print(f"{name}: normalized column aliases {rename}")
        return frame.rename(columns=rename)
    return frame


def load_parquet_checked(path: str | Path, name: str, expected: Iterable[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    frame = _canonicalize_columns(frame, name)
    _validate_columns(frame, expected, name)
    frame = _force_id_strings(frame)
    print(f"{name}: shape={frame.shape}")
    print(f"{name}: duplicate_rows={int(frame.duplicated().sum())}")
    print(f"{name}: missing_cells={int(frame.isna().sum().sum())}")
    print(f"{name}: dtypes={frame.dtypes.astype(str).to_dict()}")
    return frame


def load_all(data_dir: str | Path) -> DatasetBundle:
    """Загрузить все три файла датасета."""
    if missing_data_files(data_dir):
        raise FileNotFoundError(format_missing_data_message(data_dir))
    paths = required_paths(data_dir)
    train = load_parquet_checked(
        paths["train"], "train", QUERY_COLUMNS | ITEM_FEATURE_COLUMNS | {"item_id"}
    )
    queries = load_parquet_checked(
        paths["queries"], "benchmark_queries", QUERY_COLUMNS | {"query_id"}
    )
    items = load_parquet_checked(
        paths["items"], "benchmark_items", ITEM_FEATURE_COLUMNS | {"item_id"}
    )
    return DatasetBundle(train=train, queries=queries, items=items)


def load_train(data_dir: str | Path) -> pd.DataFrame:
    path = required_paths(data_dir)["train"]
    if not path.is_file():
        raise FileNotFoundError(format_missing_data_message(data_dir))
    return load_parquet_checked(path, "train", QUERY_COLUMNS | ITEM_FEATURE_COLUMNS | {"item_id"})


def load_benchmark(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = required_paths(data_dir)
    if not paths["queries"].is_file() or not paths["items"].is_file():
        raise FileNotFoundError(format_missing_data_message(data_dir))
    queries = load_parquet_checked(paths["queries"], "benchmark_queries", QUERY_COLUMNS | {"query_id"})
    items = load_parquet_checked(paths["items"], "benchmark_items", ITEM_FEATURE_COLUMNS | {"item_id"})
    return queries, items
