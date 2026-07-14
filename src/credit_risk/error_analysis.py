import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline

from credit_risk.config import CONFIG
from credit_risk.data.schema import TARGET_COLUMN

ERROR_KINDS = ["true_negative", "false_positive", "false_negative", "true_positive"]


def classify_predictions(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Label every applicant with the kind of outcome the model produced for them."""
    threshold = CONFIG.training.decision_threshold if threshold is None else threshold
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    outcome = pd.Series("true_negative", index=features.index, name="outcome")
    outcome[(predictions == 1) & (target == 0)] = "false_positive"
    outcome[(predictions == 0) & (target == 1)] = "false_negative"
    outcome[(predictions == 1) & (target == 1)] = "true_positive"

    return features.assign(
        **{
            TARGET_COLUMN: target,
            "probability": probabilities,
            "predicted": predictions,
            "outcome": outcome,
            "is_error": outcome.isin(["false_positive", "false_negative"]),
            # How wrong, not just whether wrong: a confident mistake is the worst kind.
            "confidence_error": np.abs(probabilities - target),
        }
    )


def error_summary(classified: pd.DataFrame) -> pd.DataFrame:
    counts = classified["outcome"].value_counts().reindex(ERROR_KINDS, fill_value=0)
    summary = pd.DataFrame({"count": counts})
    summary["share_pct"] = (summary["count"] / len(classified) * 100).round(2)
    summary["mean_probability"] = (
        classified.groupby("outcome", observed=True)["probability"]
        .mean()
        .reindex(ERROR_KINDS)
        .round(3)
    )
    return summary


def error_profile(classified: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """How the mistaken applicants differ from the ones the model got right."""
    errors = classified[classified["is_error"]]
    correct = classified[~classified["is_error"]]

    rows = {}
    for column in columns:
        if not pd.api.types.is_numeric_dtype(classified[column]):
            continue
        rows[column] = {
            "mean_when_wrong": round(errors[column].mean(), 2),
            "mean_when_right": round(correct[column].mean(), 2),
            "difference": round(errors[column].mean() - correct[column].mean(), 2),
        }
    return pd.DataFrame(rows).T.sort_values("difference", key=abs, ascending=False)


def error_rate_by_segment(classified: pd.DataFrame, column: str) -> pd.DataFrame:
    """Which segments the model fails on — the fairness and reliability question."""
    grouped = classified.groupby(column, observed=True)
    summary = pd.DataFrame(
        {
            "applicants": grouped.size(),
            "error_rate": grouped["is_error"].mean().round(3),
            "false_positive_rate": grouped["outcome"]
            .apply(lambda kinds: (kinds == "false_positive").mean())
            .round(3),
            "false_negative_rate": grouped["outcome"]
            .apply(lambda kinds: (kinds == "false_negative").mean())
            .round(3),
        }
    )
    return summary.sort_values("error_rate", ascending=False)


def confident_mistakes(classified: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """The errors the model was surest about — where its reasoning is most wrong."""
    errors = classified[classified["is_error"]]
    return errors.nlargest(count, "confidence_error")


def plot_error_overview(classified: pd.DataFrame) -> Figure:
    figure, (outcome_axis, probability_axis) = plt.subplots(1, 2, figsize=(13, 4.5))

    counts = classified["outcome"].value_counts().reindex(ERROR_KINDS, fill_value=0)
    colors = ["#b0b0b0", "#c44e52", "#dd8452", "#4c72b0"]
    outcome_axis.bar(counts.index, counts.to_numpy(), color=colors)
    outcome_axis.bar_label(outcome_axis.containers[0])
    outcome_axis.set_title("Outcomes")
    outcome_axis.tick_params(axis="x", rotation=20)

    sns.histplot(
        data=classified,
        x="probability",
        hue="is_error",
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
        ax=probability_axis,
    )
    probability_axis.set_title("Predicted probability: errors vs correct")

    figure.tight_layout()
    return figure


def plot_error_rate_by_segment(classified: pd.DataFrame, columns: list[str]) -> Figure:
    figure, axes = plt.subplots(1, len(columns), figsize=(5.5 * len(columns), 4))
    for column, axis in zip(columns, np.atleast_1d(axes), strict=True):
        rates = error_rate_by_segment(classified, column)["error_rate"]
        sns.barplot(
            x=rates.to_numpy(),
            y=rates.index.astype(str),
            hue=rates.index.astype(str),
            legend=False,
            ax=axis,
        )
        axis.axvline(classified["is_error"].mean(), ls="--", c="black", lw=1)
        axis.set(xlabel="error rate", ylabel="", title=column)
    figure.tight_layout()
    return figure
