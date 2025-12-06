import ast
from pathlib import Path

import pandas as pd
import pytest

from src import cleaning


@pytest.fixture(scope="module")
def cleaned_df():
    df = pd.read_csv("data/processed/clean.csv", parse_dates=["date", "race_time"])
    df["race_id"] = pd.to_numeric(df["race_id"], errors="coerce").astype("Int64")
    df["horse_id"] = pd.to_numeric(df["horse_id"], errors="coerce").astype("Int64")
    df["n_runners"] = pd.to_numeric(df["n_runners"], errors="coerce").astype("Int64")
    df["race_distance"] = pd.to_numeric(df["race_distance"], errors="coerce")
    return df


def test_no_obs_columns_in_cleaned_data(cleaned_df):
    obs_columns = [c for c in cleaned_df.columns if c.startswith(cleaning.OBS_PREFIX)]
    assert not obs_columns, f"obs__ columns should be dropped, found {obs_columns}"


def test_only_expected_columns(cleaned_df):
    expected = set(cleaning.NON_LEAK_COLUMNS)
    assert set(cleaned_df.columns) == expected, "Unexpected columns present in cleaned data"


def test_chronological_no_future_leakage(cleaned_df):
    ordered = cleaned_df.sort_values(["date", "race_time", "race_id"], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered, cleaned_df.reset_index(drop=True))


def test_cleaning_code_does_not_use_obs_predictors():
    """Static analysis: ensure obs__ columns are only referenced for dropping/validation."""

    source = Path("src/cleaning.py").read_text()
    tree = ast.parse(source)
    obs_usage = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("obs__"):
                obs_usage.append(node.value)
    assert set(obs_usage) == set(cleaning.SCHEMA.required_columns) - set(cleaning.NON_LEAK_COLUMNS)


def test_no_future_columns_introduced(cleaned_df):
    future_like = [c for c in cleaned_df.columns if c not in cleaning.NON_LEAK_COLUMNS]
    assert not future_like, f"Found columns that should not exist: {future_like}"
