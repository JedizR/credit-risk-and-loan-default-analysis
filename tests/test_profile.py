import pandas as pd

from credit_risk.data.schema import FEATURE_COLUMNS
from credit_risk.eda.profile import (
    approval_rate_by_band,
    categorical_summary,
    correlation_ratio,
    cramers_v,
    feature_decision_table,
    missingness,
    missingness_mechanism,
    numeric_summary,
    outlier_summary,
    overview,
    target_association,
)


def test_overview_reports_rows_and_no_duplicates(sample_frame: pd.DataFrame) -> None:
    stats = overview(sample_frame)

    assert stats["rows"] == len(sample_frame)
    assert stats["duplicate_rows"] == 0


def test_numeric_summary_exposes_missing_fraction(sample_frame: pd.DataFrame) -> None:
    summary = numeric_summary(sample_frame)

    assert summary.loc["Income", "missing_pct"] > 0
    assert summary.loc["Age", "missing_pct"] == 0


def test_categorical_summary_reports_cardinality(sample_frame: pd.DataFrame) -> None:
    summary = categorical_summary(sample_frame)

    assert summary.loc["EmploymentType", "cardinality"] == sample_frame["EmploymentType"].nunique()


def test_missingness_lists_only_missing_columns(sample_frame: pd.DataFrame) -> None:
    summary = missingness(sample_frame)

    assert set(summary.index) == {"Income", "CreditScore", "Education"}
    assert (summary["missing_count"] > 0).all()


def test_missingness_mechanism_computes_delta_and_verdict(sample_frame: pd.DataFrame) -> None:
    mechanism = missingness_mechanism(sample_frame)

    assert set(mechanism.index) == {"Income", "CreditScore", "Education"}
    assert set(mechanism["mechanism"]).issubset({"informative", "at random"})
    assert "base_rate" in mechanism.attrs


def test_outlier_summary_counts_impossible_negatives(sample_frame: pd.DataFrame) -> None:
    frame = sample_frame.copy()
    frame.loc[frame.index[0], "Income"] = -100.0

    summary = outlier_summary(frame)

    assert summary.loc["Income", "impossible_negative"] == 1
    assert summary.loc["CreditScore", "impossible_negative"] == 0


def test_cramers_v_is_high_for_identical_and_zero_for_constant() -> None:
    labels = pd.Series(["a", "b", "a", "b"] * 10)
    constant = pd.Series(["x"] * 40)

    assert cramers_v(labels, labels) > 0.9
    assert cramers_v(labels, constant) == 0.0


def test_correlation_ratio_detects_group_separation() -> None:
    categories = pd.Series(["low", "low", "high", "high"])
    separated = pd.Series([1.0, 1.0, 9.0, 9.0])
    flat = pd.Series([5.0, 5.0, 5.0, 5.0])

    assert correlation_ratio(categories, separated) > 0.9
    assert correlation_ratio(categories, flat) == 0.0


def test_target_association_ranks_creditscore_first(sample_frame: pd.DataFrame) -> None:
    association = target_association(sample_frame)

    assert association.index[0] == "CreditScore"
    assert association.loc["CreditScore", "signal"] == "signal"


def test_feature_decision_table_flags_sensitive_signal_and_impossible(
    sample_frame: pd.DataFrame,
) -> None:
    frame = sample_frame.copy()
    frame.loc[frame.index[0], "LoanAmount"] = -50.0

    table = feature_decision_table(frame)

    assert set(table.index) == set(FEATURE_COLUMNS)
    assert "exclude" in table.loc["Gender", "recommendation"]
    assert "keep" in table.loc["CreditScore", "recommendation"]
    assert "repair" in table.loc["LoanAmount", "recommendation"]


def test_decision_table_keeps_feature_with_informative_missingness(
    sample_frame: pd.DataFrame,
) -> None:
    frame = sample_frame.copy()
    frame.loc[frame["CreditScore"].isna(), "LoanApproved"] = 0

    recommendation = feature_decision_table(frame).loc["CreditScore", "recommendation"]

    assert "informative" in recommendation


def test_approval_rate_by_band_covers_every_applicant(sample_frame: pd.DataFrame) -> None:
    banded = approval_rate_by_band(sample_frame, "CreditScore", bins=4)

    assert banded["count"].sum() == sample_frame["CreditScore"].notna().sum()
    assert banded["approval_rate"].between(0, 1).all()
