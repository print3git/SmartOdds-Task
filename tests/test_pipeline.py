import pandas as pd
import pytest


def test_import_cleaning_module():
    import src.cleaning  # noqa: F401


@pytest.fixture(scope="module")
def cleaned_temp_path(tmp_path_factory):
    from src.cleaning import (
        load_raw_data,
        validate_schema,
        clean_fields,
        validate_race_invariants,
        enforce_chronological_order,
        save_cleaned_data,
    )

    temp_dir = tmp_path_factory.mktemp("cleaning_pipeline")
    output_path = temp_dir / "clean_temp.csv"

    raw_df = load_raw_data("data/raw/test_dataset.csv")
    validate_schema(raw_df)
    cleaned_df = clean_fields(raw_df)
    validate_race_invariants(cleaned_df)
    ordered_df = enforce_chronological_order(cleaned_df)
    save_cleaned_data(ordered_df, path=str(output_path))

    return output_path


def test_cleaning_runs_end_to_end(cleaned_temp_path):
    assert cleaned_temp_path.exists(), "Cleaned output file was not created"


def test_cleaned_data_basic_properties(cleaned_temp_path):
    df = pd.read_csv(cleaned_temp_path, parse_dates=["date", "race_time"])

    assert not df.empty, "Cleaned dataframe is empty"

    ordered = df.sort_values(by=["date", "race_time", "race_id"], kind="mergesort").reset_index(
        drop=True
    )
    assert ordered.equals(df.reset_index(drop=True)), "Dataframe is not sorted chronologically"

    duplicates = df.duplicated(subset=["race_id", "horse_id"])
    assert not duplicates.any(), "Duplicate (race_id, horse_id) pairs found"

    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        non_na = df[col].dropna()
        if non_na.empty:
            continue
        assert (non_na > 0).all(), f"Column {col} contains non-positive values"
