import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from credit_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from credit_risk.error_analysis import (  # noqa: E402
    ERROR_KINDS,
    classify_predictions,
    confident_mistakes,
    error_profile,
    error_rate_by_segment,
    error_summary,
    plot_error_overview,
    plot_error_rate_by_segment,
)
from credit_risk.pipeline import build_model  # noqa: E402


def _classified(sample_frame: pd.DataFrame) -> pd.DataFrame:
    features, target = sample_frame[FEATURE_COLUMNS], sample_frame[TARGET_COLUMN]
    model = build_model("logistic_regression").fit(features, target)
    return classify_predictions(model, features, target)


def test_every_applicant_gets_exactly_one_outcome(sample_frame: pd.DataFrame) -> None:
    classified = _classified(sample_frame)

    assert len(classified) == len(sample_frame)
    assert set(classified["outcome"]).issubset(ERROR_KINDS)
    assert classified["is_error"].equals(
        classified["outcome"].isin(["false_positive", "false_negative"])
    )


def test_confusion_counts_reconcile_with_the_outcomes(sample_frame: pd.DataFrame) -> None:
    classified = _classified(sample_frame)

    summary = error_summary(classified)

    assert summary["count"].sum() == len(classified)
    assert list(summary.index) == ERROR_KINDS


def test_error_profile_contrasts_wrong_and_right_applicants(sample_frame: pd.DataFrame) -> None:
    classified = _classified(sample_frame)

    profile = error_profile(classified, ["CreditScore", "Income"])

    assert set(profile.columns) == {"mean_when_wrong", "mean_when_right", "difference"}
    assert profile["difference"].abs().is_monotonic_decreasing


def test_error_rate_by_segment_covers_every_category(sample_frame: pd.DataFrame) -> None:
    classified = _classified(sample_frame)

    rates = error_rate_by_segment(classified, "EmploymentType")

    assert rates["applicants"].sum() == len(classified)
    assert rates["error_rate"].between(0, 1).all()


def test_confident_mistakes_are_errors_ranked_by_wrongness(sample_frame: pd.DataFrame) -> None:
    classified = _classified(sample_frame)

    worst = confident_mistakes(classified, count=5)

    assert worst["is_error"].all()
    assert worst["confidence_error"].is_monotonic_decreasing


def test_error_plots_return_figures(sample_frame: pd.DataFrame) -> None:
    classified = _classified(sample_frame)

    figures = [
        plot_error_overview(classified),
        plot_error_rate_by_segment(classified, ["EmploymentType", "Education"]),
    ]

    for figure in figures:
        assert isinstance(figure, Figure)
        plt.close(figure)
