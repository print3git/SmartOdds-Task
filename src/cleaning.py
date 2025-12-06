"""Data cleaning pipeline for the SmartOdds assignment.

This module implements a complete cleaning and validation workflow for the
raw race dataset. It intentionally keeps the steps small and composable so
that unit tests can exercise each stage independently. The cleaning logic is
conservative: corrupted or incoherent rows are dropped, and only pre-race
attributes are retained in the final dataset to avoid any leakage from race
outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from pathlib import Path

# Pre-define the expected schema. Categories are stored as ``object`` when
# persisted to CSV so validation focuses on dtype *kinds* rather than exact
# pandas extension types.
REQUIRED_COLUMNS: Dict[str, str] = {
    "date": "object",
    "racecourse_country": "object",
    "racecourse_name": "object",
    "race_time": "object",
    "race_id": "int",
    "race_distance": "float",
    "race_type": "object",
    "race_type_simple": "object",
    "going_clean": "object",
    "n_runners": "int",
    "horse_id": "int",
    "horse_name": "object",
    "age": "int",
    "official_rating": "int",
    "carried_weight": "float",
    "draw": "int",
    "jockey_id": "int",
    "jockey_name": "object",
    "trainer_id": "int",
    "trainer_name": "object",
    "ltp_5min": "float",
    "obs__bsp": "float",
    "obs__racing_post_rating": "float",
    "obs__uposition": "int",
    "obs__is_winner": "int",
    "obs__top_speed": "float",
    "obs__distance_to_winner": "float",
    "obs__pos_prize": "float",
    "obs__completion_time": "float",
}

# Columns that are clearly post-race observations and must be discarded to
# prevent any leakage. If the raw data contains additional ``obs__*`` fields
# they are removed wholesale in ``clean_fields``.
OBS_PREFIX = "obs" + "__"


@dataclass
class SchemaConfig:
    """Container defining the required schema and casting rules."""

    required_columns: Dict[str, str]

    @property
    def ordered_columns(self) -> List[str]:
        """List of columns in a stable order for output files."""
        return list(self.required_columns.keys())


SCHEMA = SchemaConfig(required_columns=REQUIRED_COLUMNS)

# Columns safe for modelling/evaluation that exclude any post-race observations.
NON_LEAK_COLUMNS: List[str] = [col for col in SCHEMA.ordered_columns if not col.startswith(OBS_PREFIX)]


# ---------------------------------------------------------------------------
# Loading and schema validation
# ---------------------------------------------------------------------------

def load_raw_data(path: str = "data/raw/test_dataset.csv") -> pd.DataFrame:
    """Load the raw dataset from ``path`` and enforce basic dtypes.

    Parameters
    ----------
    path:
        CSV file location. Defaults to ``data/raw/test_dataset.csv``.

    Returns
    -------
    pandas.DataFrame
        The raw dataset with dtypes coerced where possible. ``date`` and
        ``race_time`` are left as strings for explicit parsing during
        cleaning.
    """

    dtype_map = {
        "date": "string",
        "racecourse_country": "string",
        "racecourse_name": "string",
        "race_time": "string",
        "race_id": "Int64",
        "race_distance": "float64",
        "race_type": "string",
        "race_type_simple": "string",
        "going_clean": "string",
        "n_runners": "Int64",
        "horse_id": "Int64",
        "horse_name": "string",
        "age": "Int64",
        "official_rating": "Int64",
        "carried_weight": "float64",
        "draw": "Int64",
        "jockey_id": "Int64",
        "jockey_name": "string",
        "trainer_id": "Int64",
        "trainer_name": "string",
        "ltp_5min": "float64",
        "obs__bsp": "float64",
        "obs__racing_post_rating": "float64",
        "obs__uposition": "Int64",
        "obs__is_winner": "Int64",
        "obs__top_speed": "float64",
        "obs__distance_to_winner": "float64",
        "obs__pos_prize": "float64",
        "obs__completion_time": "float64",
    }

    df = pd.read_csv(path, dtype=dtype_map)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that required columns exist and have expected dtype kinds.

    Raises informative ``ValueError`` exceptions when validation fails.

    Checks performed:
    - presence of all required columns
    - correct dtype families (int/float/object/datetime)
    - no duplicate (race_id, horse_id) pairs
    """

    missing = set(SCHEMA.required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col, expected in SCHEMA.required_columns.items():
        if col not in df.columns:
            continue
        kind = df[col].dtype.kind
        if expected.startswith("datetime"):
            if not (
                pd.api.types.is_datetime64_any_dtype(df[col])
                or pd.api.types.is_object_dtype(df[col])
            ):
                raise ValueError(
                    f"Column {col} must be datetime-like or string, found {df[col].dtype}"
                )
        elif expected in {"string", "object"}:
            if not pd.api.types.is_object_dtype(df[col]) and df[col].dtype.name != "string":
                raise ValueError(f"Column {col} must be string-like, found {df[col].dtype}")
        elif expected == "Int64":
            if not (
                df[col].dtype == "Int64"
                or pd.api.types.is_integer_dtype(df[col])
                or pd.api.types.is_object_dtype(df[col])
            ):
                raise ValueError(f"Column {col} must be integer-like, found {df[col].dtype}")
        elif expected == "int":
            if not (pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                raise ValueError(f"Column {col} must be integer, found {df[col].dtype}")
        elif expected == "float":
            if not (pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                raise ValueError(f"Column {col} must be float-like, found {df[col].dtype}")

    if df.duplicated(subset=["race_id", "horse_id"]).any():
        raise ValueError("Duplicate (race_id, horse_id) pairs detected")


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _safe_numeric(series: pd.Series, dtype: str) -> pd.Series:
    """Convert a series to numeric with safe coercion and the requested dtype."""

    coerced = pd.to_numeric(series, errors="coerce")
    if dtype == "Int64":
        return coerced.astype("Int64")
    return coerced.astype(dtype)


def clean_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalise the raw dataframe.

    Steps include:
    - parsing dates and times
    - coercing numeric fields (age, draw, weight, distance)
    - removing or flagging corrupted rows
    - standardising finish positions and dropping ``obs__*`` columns
    """

    df = df.copy()

    # Parse date and time
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["race_time"] = pd.to_datetime(df["race_time"], errors="coerce")

    # Numeric conversions
    numeric_int = ["race_id", "horse_id", "n_runners", "age", "official_rating", "draw", "jockey_id", "trainer_id"]
    for col in numeric_int:
        df[col] = _safe_numeric(df[col], "Int64")

    numeric_float = [
        "race_distance",
        "carried_weight",
        "ltp_5min",
        "obs__bsp",
        "obs__racing_post_rating",
        "obs__top_speed",
        "obs__distance_to_winner",
        "obs__pos_prize",
        "obs__completion_time",
    ]
    for col in numeric_float:
        df[col] = _safe_numeric(df[col], "float64")

    # Post-race observation of finishing position
    df["obs__uposition"] = _safe_numeric(df["obs__uposition"], "Int64")
    df["obs__is_winner"] = _safe_numeric(df["obs__is_winner"], "Int64")

    # Remove rows with clearly invalid identifiers or missing essentials
    essential_cols = [
        "race_id",
        "horse_id",
        "date",
        "race_time",
        "n_runners",
        "race_distance",
    ]
    df = df.dropna(subset=essential_cols)

    # Drop impossible or invalid values
    df = df[df["race_distance"] > 0]
    df = df[df["n_runners"] > 0]
    if "age" in df:
        df = df[df["age"] > 0]

    df = df.dropna(subset=["race_id", "horse_id"])

    non_leak_columns = [col for col in SCHEMA.ordered_columns if not col.startswith(OBS_PREFIX)]
    for col in non_leak_columns:
        if col not in df.columns:
            df[col] = pd.NA

    # Ensure ordering of columns for deterministic output and remove obs__* fields
    cleaned = df[non_leak_columns]
    return cleaned


def validate_race_invariants(df: pd.DataFrame) -> None:
    """Validate race-level invariants across all rows.

    Invariants enforced for each ``race_id``:
    - date, racecourse_name, race_type_simple, and race_distance are constant
    - number of rows equals ``n_runners``
    - observed finishing positions are within ``[1, n_runners]`` or missing for non-finishers
    """

    grouped = df.groupby("race_id")
    for race_id, group in grouped:
        for field in ["date", "racecourse_name", "race_type_simple", "race_distance"]:
            if group[field].nunique(dropna=True) > 1:
                raise ValueError(f"Race {race_id} has inconsistent {field}")

        expected = group["n_runners"].iloc[0]
        if len(group) != expected:
            raise ValueError(f"Race {race_id} expected {expected} runners but found {len(group)}")

        finish_vals = group.get("obs__uposition")
        if finish_vals is None:
            continue
        finish_vals = finish_vals.dropna().astype(int)
        if not finish_vals.empty:
            if finish_vals.min() < 1 or finish_vals.max() > expected:
                raise ValueError(f"Race {race_id} has invalid finish positions")


def enforce_chronological_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort races by date, race_time, and race_id to ensure temporal order."""

    ordered = df.sort_values(by=["date", "race_time", "race_id"], kind="mergesort")
    ordered = ordered.reset_index(drop=True)
    return ordered


def save_cleaned_data(df: pd.DataFrame, path: str = "data/processed/clean.csv") -> None:
    """Persist the cleaned dataset to ``path``."""

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    df = df[[col for col in NON_LEAK_COLUMNS if col in df.columns]]
    df.to_csv(path_obj, index=False)


if __name__ == "__main__":
    raw_df = load_raw_data()
    validate_schema(raw_df)
    cleaned_df = clean_fields(raw_df)
    validate_race_invariants(cleaned_df)
    ordered_df = enforce_chronological_order(cleaned_df)
    save_cleaned_data(ordered_df)
