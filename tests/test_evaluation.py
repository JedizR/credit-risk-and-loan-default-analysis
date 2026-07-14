import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from credit_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from credit_risk.evaluation import (  # noqa: E402
    cross_validated_score,
    optimal_threshold,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_model_comparison,
    plot_roc_and_pr_curves,
    plot_threshold_cost,
    threshold_cost_curve,
)
from credit_risk.pipeline import build_model  # noqa: E402
from tests.plot_assertions import assert_figure_is_drawn  # noqa: E402


def _fitted(sample_frame: pd.DataFrame):
    model = build_model("logistic_regression")
    return model.fit(sample_frame[FEATURE_COLUMNS], sample_frame[TARGET_COLUMN])


def test_cross_validated_score_returns_mean_and_every_fold(sample_frame: pd.DataFrame) -> None:
    scores = cross_validated_score(
        sample_frame[FEATURE_COLUMNS], sample_frame[TARGET_COLUMN], "logistic_regression"
    )

    assert 0.0 <= scores["mean"] <= 1.0
    assert len(scores["folds"]) == 5


def test_threshold_cost_curve_prices_both_kinds_of_mistake(sample_frame: pd.DataFrame) -> None:
    model = _fitted(sample_frame)
    probabilities = model.predict_proba(sample_frame[FEATURE_COLUMNS])[:, 1]

    costs = threshold_cost_curve(sample_frame[TARGET_COLUMN], probabilities)

    assert set(costs.columns) == {
        "threshold",
        "false_approvals",
        "false_rejections",
        "expected_cost",
    }
    assert (costs["expected_cost"] >= 0).all()


def test_optimal_threshold_minimises_expected_cost(sample_frame: pd.DataFrame) -> None:
    model = _fitted(sample_frame)
    probabilities = model.predict_proba(sample_frame[FEATURE_COLUMNS])[:, 1]

    best = optimal_threshold(sample_frame[TARGET_COLUMN], probabilities)
    costs = threshold_cost_curve(sample_frame[TARGET_COLUMN], probabilities)

    assert (
        costs.loc[costs["threshold"] == best, "expected_cost"].iloc[0]
        == costs["expected_cost"].min()
    )


def test_costlier_false_approvals_raise_the_threshold(sample_frame: pd.DataFrame) -> None:
    model = _fitted(sample_frame)
    probabilities = model.predict_proba(sample_frame[FEATURE_COLUMNS])[:, 1]
    target = sample_frame[TARGET_COLUMN]

    cautious = optimal_threshold(
        target, probabilities, false_approval_cost=20.0, false_rejection_cost=1.0
    )
    lenient = optimal_threshold(
        target, probabilities, false_approval_cost=1.0, false_rejection_cost=20.0
    )

    assert cautious > lenient


def test_curve_plots_return_figures(sample_frame: pd.DataFrame) -> None:
    model = _fitted(sample_frame)
    features, target = sample_frame[FEATURE_COLUMNS], sample_frame[TARGET_COLUMN]
    probabilities = model.predict_proba(features)[:, 1]

    figures = [
        plot_roc_and_pr_curves({"logreg": model}, features, target),
        plot_calibration_curve({"logreg": model}, features, target),
        plot_confusion_matrix(model, features, target),
        plot_threshold_cost(target, probabilities),
    ]

    for figure in figures:
        assert_figure_is_drawn(figure)
        plt.close(figure)


def test_learning_curve_plot_returns_a_figure(sample_frame: pd.DataFrame) -> None:
    figure = plot_learning_curve(
        sample_frame[FEATURE_COLUMNS], sample_frame[TARGET_COLUMN], "logistic_regression"
    )

    assert_figure_is_drawn(figure)
    plt.close(figure)


def test_model_comparison_plot_returns_a_figure() -> None:
    scores = pd.DataFrame({"mean": [0.8, 0.9], "std": [0.01, 0.02]}, index=["a", "b"])

    figure = plot_model_comparison(scores)

    assert_figure_is_drawn(figure)
    plt.close(figure)
