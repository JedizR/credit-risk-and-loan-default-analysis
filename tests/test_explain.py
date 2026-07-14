import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from credit_risk.data.schema import TARGET_COLUMN  # noqa: E402
from credit_risk.explain import (  # noqa: E402
    Explanation,
    decision_reasons,
    explain_model,
    global_importance,
    local_contributions,
    plot_beeswarm,
    plot_dependence,
    plot_importance_bar,
)
from credit_risk.features.engineering import engineer_features  # noqa: E402
from credit_risk.features.selection import candidate_features, split_feature_types  # noqa: E402
from credit_risk.pipeline import build_model  # noqa: E402


def _fitted(sample_frame: pd.DataFrame):
    engineered = engineer_features(sample_frame)
    columns = candidate_features()
    numeric, categorical = split_feature_types(columns)
    model = build_model("lightgbm", numeric, categorical)
    features = engineered[columns]
    model.fit(features, engineered[TARGET_COLUMN])
    return model, features


def test_explanation_has_one_value_per_row_and_encoded_feature(sample_frame: pd.DataFrame) -> None:
    model, features = _fitted(sample_frame)

    explanation = explain_model(model, features)

    assert isinstance(explanation, Explanation)
    assert explanation.values.shape[0] == len(features)
    assert explanation.values.shape[1] == len(explanation.feature_names)


def test_global_importance_is_ranked_and_non_negative(sample_frame: pd.DataFrame) -> None:
    model, features = _fitted(sample_frame)

    importance = global_importance(explain_model(model, features))

    assert (importance >= 0).all()
    assert importance.is_monotonic_decreasing


def test_local_contributions_are_sorted_by_magnitude(sample_frame: pd.DataFrame) -> None:
    model, features = _fitted(sample_frame)

    contributions = local_contributions(explain_model(model, features), position=0)

    assert contributions["shap"].abs().is_monotonic_decreasing
    assert set(contributions["direction"]).issubset({"raises", "lowers"})


def test_reasons_are_human_readable_and_signed(sample_frame: pd.DataFrame) -> None:
    model, features = _fitted(sample_frame)

    reasons = decision_reasons(model, features.head(20))

    assert set(reasons.columns) == {"decision", "probability", "reasons"}
    assert reasons["decision"].isin(["approved", "declined"]).all()
    assert reasons["probability"].between(0, 1).all()
    # Reasons name features, never raw column codes or floats.
    assert not reasons["reasons"].str.contains("__").any()
    assert reasons["reasons"].str.contains(r"\(\+\)|\(-\)").all()


def test_reason_count_is_capped(sample_frame: pd.DataFrame) -> None:
    model, features = _fitted(sample_frame)

    reasons = decision_reasons(model, features.head(5), top_reasons=2)

    assert (reasons["reasons"].str.count(";") == 1).all()


def test_shap_plots_are_drawn_not_blank(sample_frame: pd.DataFrame) -> None:
    model, features = _fitted(sample_frame)
    explanation = explain_model(model, features)

    figures = [
        plot_beeswarm(explanation),
        plot_importance_bar(explanation),
        plot_dependence(explanation, "CreditScore", "is_unemployed"),
    ]

    for figure in figures:
        assert isinstance(figure, Figure)
        # A figure object alone proves nothing: shap builds its own figures, so assert that
        # something was actually drawn on the one that comes back.
        assert figure.axes
        assert any(axis.collections or axis.lines or axis.patches for axis in figure.axes)
        plt.close(figure)
