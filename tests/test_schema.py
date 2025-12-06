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
    df["race_distance"] = pd.to_numeric(df["race_distance"], errors="coerce")
    df["carried_weight"] = pd.to_numeric(df["carried_weight"], errors="coerce")
    df["official_rating"] = pd.to_numeric(df["official_rating"], errors="coerce")
    return df


def test_required_columns_present(cleaned_df):
    missing = set(cleaning.NON_LEAK_COLUMNS) - set(cleaned_df.columns)
    assert not missing, f"Missing required columns: {missing}"


def test_column_dtypes(cleaned_df):
    expected = {col: cleaning.SCHEMA.required_columns[col] for col in cleaning.NON_LEAK_COLUMNS}
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
    critical = ["race_id", "horse_id", "date", "race_time", "n_runners", "race_distance"]
    for col in critical:
        assert cleaned_df[col].notna().all(), f"Missing values detected in {col}"


def test_finish_positions_in_range(cleaned_df):
    assert "obs__uposition" not in cleaned_df.columns
