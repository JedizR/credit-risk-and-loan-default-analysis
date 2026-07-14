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

DTYPES = {
    "Age": "int64",
    "Income": "float64",
    "LoanAmount": "float64",
    "CreditScore": "float64",
    "YearsExperience": "int64",
    "Gender": "category",
    "Education": "category",
    "City": "category",
    "EmploymentType": "category",
    "LoanApproved": "int64",
}


class MissingColumnsError(ValueError):
    def __init__(self, source: object, missing_columns: list[str]) -> None:
        super().__init__(f"{source} is missing required columns: {', '.join(missing_columns)}")


def require_columns(frame: pd.DataFrame, required_columns: list[str], source: object) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise MissingColumnsError(source, missing_columns)


def apply_schema_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    present_dtypes = {column: dtype for column, dtype in DTYPES.items() if column in frame.columns}
    return frame.astype(present_dtypes)
