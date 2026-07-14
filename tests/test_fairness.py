import matplotlib
import pandas as pd

matplotlib.use("Agg")

from credit_risk.config import CONFIG  # noqa: E402
from credit_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from credit_risk.error_analysis import classify_predictions  # noqa: E402
from credit_risk.fairness import (  # noqa: E402
    FOUR_FIFTHS,
    disparate_impact,
    fairness_audit,
    impact_ratio_confidence,
    plot_fairness,
)
from credit_risk.pipeline import build_model  # noqa: E402
from tests.plot_assertions import assert_figure_is_drawn  # noqa: E402


def _classified(sample_frame: pd.DataFrame) -> pd.DataFrame:
    features, target = sample_frame[FEATURE_COLUMNS], sample_frame[TARGET_COLUMN]
    model = build_model("logistic_regression").fit(features, target)
    return classify_predictions(model, features, target)


def test_audit_covers_every_protected_group(sample_frame: pd.DataFrame) -> None:
    audit = fairness_audit(_classified(sample_frame))

    attributes = audit.index.get_level_values("attribute").unique()
    assert set(attributes) == set(CONFIG.sensitive_features)
    assert audit["applicants"].sum() == len(sample_frame) * len(CONFIG.sensitive_features)


def test_audit_reports_the_metrics_a_regulator_asks_for(sample_frame: pd.DataFrame) -> None:
    audit = fairness_audit(_classified(sample_frame))

    assert set(audit.columns) == {
        "applicants",
        "selection_rate",
        "error_rate",
        "false_positive_rate",
        "false_negative_rate",
    }
    assert audit["selection_rate"].between(0, 1).all()


def test_disparate_impact_applies_the_four_fifths_rule(sample_frame: pd.DataFrame) -> None:
    impact = disparate_impact(fairness_audit(_classified(sample_frame)))

    assert impact["impact_ratio"].between(0, 1).all()
    for attribute in impact.index:
        expected = impact.loc[attribute, "impact_ratio"] >= FOUR_FIFTHS
        assert impact.loc[attribute, "passes_four_fifths"] == expected


def test_bootstrap_reports_an_ordered_interval(sample_frame: pd.DataFrame) -> None:
    result = impact_ratio_confidence(_classified(sample_frame), "Gender", resamples=100)

    # A min/max ratio is a biased statistic under resampling, so the interval is not guaranteed
    # to bracket the point estimate — only to be ordered and to live on the ratio scale.
    assert 0.0 <= result["lower_95"] <= result["upper_95"] <= 1.0
    assert result["passes_four_fifths"] == (result["impact_ratio"] >= FOUR_FIFTHS)
    # Bias can only be *concluded* when the whole interval sits below the threshold.
    assert result["conclusive"] == (result["upper_95"] < FOUR_FIFTHS)


def test_fairness_plot_is_drawn(sample_frame: pd.DataFrame) -> None:
    figure = plot_fairness(fairness_audit(_classified(sample_frame)))

    assert_figure_is_drawn(figure)
