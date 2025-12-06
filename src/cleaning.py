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
from typing import Dict, Iterable, List

import pandas as pd
from pathlib import Path

# Pre-define the expected schema. Categories are stored as ``object`` when
# persisted to CSV so validation focuses on dtype *kinds* rather than exact
# pandas extension types.
REQUIRED_COLUMNS: Dict[str, str] = {
    "race_id": "int",
    "horse_id": "int",
    "date": "datetime64[ns]",
    "race_time": "datetime64[ns]",
    "racecourse": "object",
    "race_type_simple": "object",
    "distance": "float",
    "n_runners": "int",
    "draw": "float",
    "age": "float",
    "weight_lbs": "float",
    "finish_position": "Int64",
}

# Columns that are clearly post-race observations and must be discarded to
# prevent any leakage. If the raw data contains additional ``obs__*`` fields
# they are removed wholesale in ``clean_fields``.
OBS_PREFIX = "obs__"


@dataclass
class SchemaConfig:
    """Container defining the required schema and casting rules."""

    required_columns: Dict[str, str]

    @property
    def ordered_columns(self) -> List[str]:
        """List of columns in a stable order for output files."""
        return list(self.required_columns.keys())


SCHEMA = SchemaConfig(required_columns=REQUIRED_COLUMNS)


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
        "race_id": "int64",
        "horse_id": "int64",
        "racecourse": "string",
        "race_type_simple": "string",
        "distance": "float64",
        "n_runners": "Int64",
        "draw": "float64",
        "age": "float64",
        "weight_lbs": "float64",
        # obs__ columns are not coerced here intentionally.
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
        elif expected == "object":
            if kind not in ("O", "U", "S"):
                raise ValueError(f"Column {col} must be object-like, found {df[col].dtype}")

    if df.duplicated(subset=["race_id", "horse_id"]).any():
        raise ValueError("Duplicate (race_id, horse_id) pairs detected")


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _parse_weight(value: object) -> float | None:
    """Parse weights expressed as pounds or ``stones-pounds`` strings."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if "-" in text:
        try:
            stones, pounds = text.split("-")
            return float(stones) * 14 + float(pounds)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_distance(value: object) -> float | None:
    """Convert race distance strings to yards where possible."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    # Support composite distances like "2m4f110y"
    yards = 0.0
    num = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            num += ch
            continue
        if ch in {"m", "f", "y"}:
            try:
                magnitude = float(num)
            except ValueError:
                magnitude = 0.0
            if ch == "m":
                yards += magnitude * 1760
            elif ch == "f":
                yards += magnitude * 220
            elif ch == "y":
                yards += magnitude
            num = ""
    if num:
        try:
            yards += float(num)
        except ValueError:
            pass
    return yards if yards > 0 else None


def _standardize_finish(value: object) -> float | None:
    """Normalise finish positions, mapping non-finishers to ``None``."""
    if pd.isna(value):
        return None
    non_finish_codes: Iterable[str] = {
        "pu",
        "ur",
        "f",
        "bd",
        "ro",
        "ref",
        "voi",
        "lft",
        "su",
        "dsq",
        "dnf",
        "ot",
        "otd",
        "bf",
    }
    text = str(value).strip().lower()
    if text in non_finish_codes:
        return None
    try:
        num = float(text)
        return num if num > 0 else None
    except ValueError:
        return None


def clean_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalise the raw dataframe.

    Steps include:
    - parsing dates and times
    - coercing numeric fields (age, draw, weight, distance)
    - removing or flagging corrupted rows
    - standardising finish positions and dropping ``obs__*`` columns
    """

    df = df.copy()

    # Drop all obs__* columns to prevent leakage, but capture finish position
    finish_source = None
    for col in list(df.columns):
        if col.startswith(OBS_PREFIX):
            if col == "obs__finish_position" and finish_source is None:
                finish_source = col
                continue
            df = df.drop(columns=col)

    if finish_source is None and "finish_position" not in df.columns:
        raise ValueError("Finish position column missing; expected obs__finish_position or finish_position")

    # Parse date and time
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["race_time"] = pd.to_datetime(df["race_time"], errors="coerce")

    # Numeric conversions
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["draw"] = pd.to_numeric(df["draw"], errors="coerce")
    df["weight_lbs"] = df["weight_lbs"].apply(_parse_weight)
    df["weight_lbs"] = pd.to_numeric(df["weight_lbs"], errors="coerce")
    df["distance"] = df["distance"].apply(_parse_distance)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["n_runners"] = pd.to_numeric(df["n_runners"], errors="coerce").astype("Int64")

    # Finish positions
    if finish_source:
        finish_values = df[finish_source]
    else:
        finish_values = df["finish_position"]
    df["finish_position"] = finish_values.apply(_standardize_finish).astype("Float64")
    df["finish_position"] = df["finish_position"].round().astype("Int64")
    if finish_source:
        df = df.drop(columns=finish_source)

    # Remove rows with clearly invalid identifiers or missing essentials
    essential_cols = ["race_id", "horse_id", "date", "race_time", "n_runners", "distance"]
    df = df.dropna(subset=essential_cols)

    # Drop impossible or invalid values
    df = df[df["age"] > 0]
    df = df[df["distance"] > 0]
    df = df[df["n_runners"] > 0]

    # Cast identifiers to integer
    df["race_id"] = pd.to_numeric(df["race_id"], errors="coerce").astype("Int64")
    df["horse_id"] = pd.to_numeric(df["horse_id"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["race_id", "horse_id"])

    # Ensure ordering of columns for deterministic output
    for col in SCHEMA.ordered_columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[SCHEMA.ordered_columns]


def validate_race_invariants(df: pd.DataFrame) -> None:
    """Validate race-level invariants across all rows.

    Invariants enforced for each ``race_id``:
    - date, racecourse, race_type_simple, and distance are constant
    - number of rows equals ``n_runners``
    - finish positions are within ``[1, n_runners]`` or missing for non-finishers
    """

    grouped = df.groupby("race_id")
    for race_id, group in grouped:
        for field in ["date", "racecourse", "race_type_simple", "distance"]:
            if group[field].nunique(dropna=True) > 1:
                raise ValueError(f"Race {race_id} has inconsistent {field}")

        expected = group["n_runners"].iloc[0]
        if len(group) != expected:
            raise ValueError(f"Race {race_id} expected {expected} runners but found {len(group)}")

        finish_vals = group["finish_position"].dropna().astype(int)
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
    df.to_csv(path_obj, index=False)


if __name__ == "__main__":
    raw_df = load_raw_data()
    validate_schema(raw_df)
    cleaned_df = clean_fields(raw_df)
    validate_race_invariants(cleaned_df)
    ordered_df = enforce_chronological_order(cleaned_df)
    save_cleaned_data(ordered_df)
