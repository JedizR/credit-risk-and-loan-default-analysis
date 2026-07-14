import numpy as np
import pandas as pd

from credit_risk.config import CONFIG
from credit_risk.features.engineering import (
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    FeatureEngineer,
    engineer_features,
)


def test_every_engineered_column_is_produced(sample_frame: pd.DataFrame) -> None:
    engineered = engineer_features(sample_frame)

    assert set(ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL).issubset(engineered.columns)
    assert len(engineered) == len(sample_frame)


def test_missingness_indicators_match_the_source_columns(sample_frame: pd.DataFrame) -> None:
    engineered = engineer_features(sample_frame)

    assert engineered["credit_score_missing"].sum() == sample_frame["CreditScore"].isna().sum()
    assert engineered["income_missing"].sum() == sample_frame["Income"].isna().sum()


def test_interaction_zeroes_out_for_unemployed_applicants(sample_frame: pd.DataFrame) -> None:
    frame = sample_frame.copy()
    frame.loc[frame.index[0], "EmploymentType"] = "Unemployed"
    frame.loc[frame.index[0], "CreditScore"] = 800.0

    engineered = engineer_features(frame)

    assert engineered.loc[frame.index[0], "credit_score_x_employed"] == 0
    assert engineered.loc[frame.index[0], "is_unemployed"] == 1


def test_thresholds_come_from_config(sample_frame: pd.DataFrame) -> None:
    engineered = engineer_features(sample_frame)

    expected = (sample_frame["CreditScore"] < CONFIG.thresholds.low_credit_score).sum()
    assert engineered["low_credit_score"].sum() == expected


def test_debt_to_income_never_divides_by_zero() -> None:
    frame = pd.DataFrame(
        {
            "Age": [30],
            "Income": [0.0],
            "LoanAmount": [1000.0],
            "CreditScore": [700.0],
            "YearsExperience": [5],
            "Gender": ["Male"],
            "Education": ["PhD"],
            "City": ["Chicago"],
            "EmploymentType": ["Salaried"],
        }
    )

    engineered = engineer_features(frame)

    assert np.isfinite(engineered["debt_to_income"]).all()


def test_engineering_is_stateless_so_it_cannot_leak(sample_frame: pd.DataFrame) -> None:
    first_half = sample_frame.iloc[:100]
    engineer = FeatureEngineer().fit(sample_frame)

    fitted_on_all = engineer.transform(first_half)
    fitted_on_nothing = FeatureEngineer().transform(first_half)

    pd.testing.assert_frame_equal(fitted_on_all, fitted_on_nothing)


def test_credit_band_is_ordered_by_score(sample_frame: pd.DataFrame) -> None:
    engineered = engineer_features(sample_frame)
    banded = engineered.dropna(subset=["CreditScore", "credit_band"])

    poor = banded.loc[banded["credit_band"] == "poor", "CreditScore"]
    prime = banded.loc[banded["credit_band"] == "prime", "CreditScore"]

    assert poor.max() < prime.min()
