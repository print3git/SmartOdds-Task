import pandas as pd
import pytest

from src import cleaning


@pytest.fixture(scope="module")
def cleaned_df():
    df = pd.read_csv("data/processed/clean.csv", parse_dates=["date", "race_time"])
    df["race_id"] = pd.to_numeric(df["race_id"], errors="coerce").astype("Int64")
    df["horse_id"] = pd.to_numeric(df["horse_id"], errors="coerce").astype("Int64")
    df["n_runners"] = pd.to_numeric(df["n_runners"], errors="coerce").astype("Int64")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce").astype("Int64")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["draw"] = pd.to_numeric(df["draw"], errors="coerce")
    df["weight_lbs"] = pd.to_numeric(df["weight_lbs"], errors="coerce")
    return df


def test_race_level_invariants(cleaned_df):
    grouped = cleaned_df.groupby("race_id")
    for race_id, group in grouped:
        assert group["date"].nunique(dropna=True) == 1, f"Date mismatch in race {race_id}"
        assert group["racecourse"].nunique(dropna=True) == 1, f"Racecourse mismatch in race {race_id}"
        assert group["race_type_simple"].nunique(dropna=True) == 1, f"Type mismatch in race {race_id}"
        assert group["distance"].nunique(dropna=True) == 1, f"Distance mismatch in race {race_id}"


def test_runner_count_matches(cleaned_df):
    grouped = cleaned_df.groupby("race_id")
    for race_id, group in grouped:
        expected = int(group["n_runners"].iloc[0])
        assert len(group) == expected, f"Race {race_id} expected {expected} runners"


def test_numeric_fields_positive(cleaned_df):
    numeric_positive = ["distance", "age", "weight_lbs", "draw"]
    for col in numeric_positive:
        assert (cleaned_df[col] > 0).all(), f"Non-positive values found in {col}"
    assert (cleaned_df["n_runners"] > 0).all(), "n_runners must be positive"


def test_chronological_order(cleaned_df):
    ordered = cleaning.enforce_chronological_order(cleaned_df)
    pd.testing.assert_frame_equal(ordered, cleaned_df.reset_index(drop=True))
