from pathlib import Path

import pandas as pd
import pytest

from credit_risk.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MissingColumnsError,
    load_features_to_score,
    load_training_data,
)


def test_load_training_data_keeps_features_and_target(training_csv: Path) -> None:
    frame = load_training_data(training_csv)

    assert TARGET_COLUMN in frame.columns
    assert set(FEATURE_COLUMNS).issubset(frame.columns)


def test_load_features_to_score_drops_unexpected_columns(
    sample_frame: pd.DataFrame, tmp_path: Path
) -> None:
    csv_path = tmp_path / "extra.csv"
    sample_frame.drop(columns=[TARGET_COLUMN]).assign(ApplicantId="A-1").to_csv(
        csv_path, index=False
    )

    frame = load_features_to_score(csv_path)

    assert list(frame.columns) == FEATURE_COLUMNS


def test_load_training_data_rejects_missing_target(
    sample_frame: pd.DataFrame, tmp_path: Path
) -> None:
    csv_path = tmp_path / "no_target.csv"
    sample_frame.drop(columns=[TARGET_COLUMN]).to_csv(csv_path, index=False)

    with pytest.raises(MissingColumnsError, match=TARGET_COLUMN):
        load_training_data(csv_path)


def test_load_features_to_score_rejects_missing_feature(
    sample_frame: pd.DataFrame, tmp_path: Path
) -> None:
    csv_path = tmp_path / "no_income.csv"
    sample_frame.drop(columns=["Income"]).to_csv(csv_path, index=False)

    with pytest.raises(MissingColumnsError, match="Income"):
        load_features_to_score(csv_path)
