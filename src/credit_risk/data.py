from pathlib import Path

import pandas as pd

TARGET_COLUMN = "LoanApproved"

NUMERIC_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "YearsExperience",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Education",
    "City",
    "EmploymentType",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class MissingColumnsError(ValueError):
    def __init__(self, csv_path: Path, missing_columns: list[str]) -> None:
        super().__init__(f"{csv_path} is missing required columns: {', '.join(missing_columns)}")


def _read_csv_with_required_columns(csv_path: Path, required_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise MissingColumnsError(csv_path, missing_columns)
    return frame


def load_training_data(csv_path: Path) -> pd.DataFrame:
    return _read_csv_with_required_columns(csv_path, FEATURE_COLUMNS + [TARGET_COLUMN])


def load_features_to_score(csv_path: Path) -> pd.DataFrame:
    frame = _read_csv_with_required_columns(csv_path, FEATURE_COLUMNS)
    return frame[FEATURE_COLUMNS]
