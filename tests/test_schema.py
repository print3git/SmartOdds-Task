import pandas as pd
import pytest

from src import cleaning


@pytest.fixture(scope="module")
def cleaned_df():
    path = "data/processed/clean.csv"
    df = pd.read_csv(path, parse_dates=["date", "race_time"])
    # Coerce types to expected pandas representations
    df["race_id"] = pd.to_numeric(df["race_id"], errors="coerce").astype("Int64")
    df["horse_id"] = pd.to_numeric(df["horse_id"], errors="coerce").astype("Int64")
    df["n_runners"] = pd.to_numeric(df["n_runners"], errors="coerce").astype("Int64")
    df["draw"] = pd.to_numeric(df["draw"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["weight_lbs"] = pd.to_numeric(df["weight_lbs"], errors="coerce")
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce").astype("Int64")
    return df


def test_required_columns_present(cleaned_df):
    missing = set(cleaning.SCHEMA.required_columns) - set(cleaned_df.columns)
    assert not missing, f"Missing required columns: {missing}"


def test_column_dtypes(cleaned_df):
    expected = cleaning.SCHEMA.required_columns
    for col, dtype_kind in expected.items():
        if dtype_kind.startswith("datetime"):
            assert pd.api.types.is_datetime64_any_dtype(cleaned_df[col])
        elif dtype_kind in {"int", "Int64"}:
            assert pd.api.types.is_integer_dtype(cleaned_df[col])
        elif dtype_kind == "float":
            assert pd.api.types.is_float_dtype(cleaned_df[col])
        elif dtype_kind == "object":
            assert cleaned_df[col].dtype.kind in ("O", "U", "S")


def test_no_duplicate_runner_keys(cleaned_df):
    duplicates = cleaned_df.duplicated(subset=["race_id", "horse_id"])
    assert not duplicates.any(), "Duplicate (race_id, horse_id) pairs found"


def test_no_missing_critical_fields(cleaned_df):
    critical = ["race_id", "horse_id", "date", "race_time", "finish_position"]
    for col in critical:
        assert cleaned_df[col].notna().all(), f"Missing values detected in {col}"


def test_finish_positions_in_range(cleaned_df):
    grouped = cleaned_df.groupby("race_id")
    for race_id, group in grouped:
        n_runners = int(group["n_runners"].iloc[0])
        finishes = group["finish_position"].dropna().astype(int)
        if finishes.empty:
            continue
        assert finishes.min() >= 1, f"Race {race_id} has finish < 1"
        assert finishes.max() <= n_runners, f"Race {race_id} finish exceeds n_runners"
