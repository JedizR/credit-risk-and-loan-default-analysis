from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import TARGET_COLUMN

SAMPLE_SIZE = 300


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    generator = np.random.default_rng(seed=7)
    frame = pd.DataFrame(
        {
            "Age": generator.integers(21, 70, SAMPLE_SIZE),
            "Income": generator.normal(50_000, 12_000, SAMPLE_SIZE),
            "LoanAmount": generator.normal(25_000, 8_000, SAMPLE_SIZE),
            "CreditScore": generator.normal(650, 70, SAMPLE_SIZE),
            "YearsExperience": generator.integers(0, 40, SAMPLE_SIZE),
            "Gender": generator.choice(["Male", "Female"], SAMPLE_SIZE),
            "Education": generator.choice(["High School", "Bachelors", "Masters"], SAMPLE_SIZE),
            "City": generator.choice(["Chicago", "Houston", "New York"], SAMPLE_SIZE),
            "EmploymentType": generator.choice(["Salaried", "Self-Employed"], SAMPLE_SIZE),
        }
    )
    approval_signal = frame["CreditScore"] + generator.normal(0, 30, SAMPLE_SIZE)
    frame[TARGET_COLUMN] = (approval_signal > approval_signal.quantile(0.75)).astype(int)

    frame.loc[frame.index[:15], "Income"] = np.nan
    frame.loc[frame.index[15:30], "CreditScore"] = np.nan
    frame.loc[frame.index[30:45], "Education"] = np.nan
    return frame


@pytest.fixture
def training_csv(sample_frame: pd.DataFrame, tmp_path: Path) -> Path:
    csv_path = tmp_path / "training.csv"
    sample_frame.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def scoring_csv(sample_frame: pd.DataFrame, tmp_path: Path) -> Path:
    csv_path = tmp_path / "applicants.csv"
    sample_frame.drop(columns=[TARGET_COLUMN]).to_csv(csv_path, index=False)
    return csv_path
