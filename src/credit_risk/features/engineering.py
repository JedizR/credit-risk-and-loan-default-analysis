import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.config import CONFIG
from credit_risk.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

CREDIT_BANDS = [0, 500, 580, 670, 740, 850]
CREDIT_BAND_LABELS = ["poor", "fair", "near_prime", "prime", "super_prime"]

ENGINEERED_NUMERIC = [
    "debt_to_income",
    "experience_per_age",
    "credit_score_x_employed",
    "income_x_employed",
    "credit_score_missing",
    "income_missing",
    "is_unemployed",
    "low_credit_score",
    "low_income",
]
ENGINEERED_CATEGORICAL = ["credit_band"]

MODEL_NUMERIC_FEATURES = NUMERIC_FEATURES + ENGINEERED_NUMERIC
MODEL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds missingness indicators, domain interactions, ratios and bands.

    Stateless by design: nothing is learned from the data, so applying it before the
    train/holdout split cannot leak. Everything that *is* learned (imputation, scaling,
    encoding) stays inside the model pipeline.
    """

    def fit(self, features: pd.DataFrame, target: pd.Series | None = None) -> "FeatureEngineer":  # noqa: ARG002
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add the engineered columns to a copy of the frame.

        Each engineered feature encodes something the exploratory analysis found:

        - ``low_credit_score`` folds a *missing* credit score into the low-score flag: those
          applicants approve at 2.1%, below even the worst observed band, so a missing score is bad
          news rather than "unknown".
        - ``credit_score_x_employed`` and ``income_x_employed`` encode the soft AND — a good score
          or income only pays off while the applicant is employed.
        - ``debt_to_income`` is capped, because a handful of near-zero incomes would otherwise send
          the ratio into the hundreds and dominate the scaler; past the cap the applicant is already
          maximally indebted.
        - ``credit_band`` stays a category so it survives a parquet round-trip with the same dtype
          as the source categoricals.
        """
        engineered = features.copy()

        engineered["credit_score_missing"] = features["CreditScore"].isna().astype(int)
        engineered["income_missing"] = features["Income"].isna().astype(int)

        employed = features["EmploymentType"].ne("Unemployed").astype(int)
        engineered["is_unemployed"] = 1 - employed

        engineered["low_credit_score"] = (
            features["CreditScore"].isna()
            | (features["CreditScore"] < CONFIG.thresholds.low_credit_score)
        ).astype(int)
        engineered["low_income"] = (features["Income"] < CONFIG.thresholds.low_income).astype(int)

        engineered["credit_score_x_employed"] = features["CreditScore"] * employed
        engineered["income_x_employed"] = features["Income"] * employed

        engineered["debt_to_income"] = (
            features["LoanAmount"] / features["Income"].clip(lower=1.0)
        ).clip(upper=CONFIG.thresholds.max_debt_to_income)
        engineered["experience_per_age"] = features["YearsExperience"] / features["Age"].replace(
            0, np.nan
        )

        engineered["credit_band"] = pd.cut(
            features["CreditScore"], bins=CREDIT_BANDS, labels=CREDIT_BAND_LABELS
        ).astype("category")

        return engineered

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:  # noqa: ARG002
        return np.asarray(MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the engineered columns, keeping any non-feature columns (such as the target)."""
    return FeatureEngineer().transform(frame)
