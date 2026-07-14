from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    learning_curve,
)
from sklearn.pipeline import Pipeline

from credit_risk.config import CONFIG
from credit_risk.pipeline import DEFAULT_MODEL_NAME, build_model

SCORING = "average_precision"


def cross_validation_folds() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=CONFIG.training.cross_validation_folds,
        shuffle=True,
        random_state=CONFIG.seed,
    )


def cross_validated_score(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str = DEFAULT_MODEL_NAME,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Mean cross-validated PR-AUC. Preprocessing is refit inside every fold, so it cannot leak."""
    model = build_model(model_name, numeric_features, categorical_features, params)
    scores = cross_val_score(
        model, features, target, scoring=SCORING, cv=cross_validation_folds(), n_jobs=-1
    )
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "folds": [float(score) for score in scores],
    }


def threshold_cost_curve(
    target: pd.Series,
    probabilities: np.ndarray,
    false_approval_cost: float | None = None,
    false_rejection_cost: float | None = None,
) -> pd.DataFrame:
    """Expected cost of every decision threshold, given what each kind of mistake costs."""
    false_approval_cost = false_approval_cost or CONFIG.training.false_approval_cost
    false_rejection_cost = false_rejection_cost or CONFIG.training.false_rejection_cost

    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = (probabilities >= threshold).astype(int)
        false_approvals = int(((predictions == 1) & (target == 0)).sum())
        false_rejections = int(((predictions == 0) & (target == 1)).sum())
        rows.append(
            {
                "threshold": round(float(threshold), 3),
                "false_approvals": false_approvals,
                "false_rejections": false_rejections,
                "expected_cost": (
                    false_approvals * false_approval_cost + false_rejections * false_rejection_cost
                ),
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(
    target: pd.Series,
    probabilities: np.ndarray,
    false_approval_cost: float | None = None,
    false_rejection_cost: float | None = None,
) -> float:
    costs = threshold_cost_curve(target, probabilities, false_approval_cost, false_rejection_cost)
    return float(costs.loc[costs["expected_cost"].idxmin(), "threshold"])


def plot_roc_and_pr_curves(
    models: dict[str, Pipeline], features: pd.DataFrame, target: pd.Series
) -> Figure:
    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(13, 5))
    base_rate = target.mean()

    for name, model in models.items():
        probabilities = model.predict_proba(features)[:, 1]

        false_positive_rate, true_positive_rate, _ = roc_curve(target, probabilities)
        roc_axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC={roc_auc_score(target, probabilities):.3f})",
        )

        precision, recall, _ = precision_recall_curve(target, probabilities)
        pr_axis.plot(
            recall,
            precision,
            label=f"{name} (AP={average_precision_score(target, probabilities):.3f})",
        )

    roc_axis.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    roc_axis.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    roc_axis.legend(loc="lower right")

    pr_axis.axhline(base_rate, ls="--", c="k", lw=1, label=f"base rate ({base_rate:.0%})")
    pr_axis.set(xlabel="recall", ylabel="precision", title="Precision-Recall")
    pr_axis.legend(loc="lower left")

    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_calibration_curve(
    models: dict[str, Pipeline], features: pd.DataFrame, target: pd.Series, bins: int = 10
) -> Figure:
    figure, axis = plt.subplots(figsize=(6.5, 6))
    for name, model in models.items():
        probabilities = model.predict_proba(features)[:, 1]
        observed, predicted = calibration_curve(
            target, probabilities, n_bins=bins, strategy="quantile"
        )
        axis.plot(predicted, observed, marker="o", label=name)

    axis.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    axis.set(
        xlabel="predicted approval probability",
        ylabel="observed approval rate",
        title="Calibration",
    )
    axis.legend()
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_confusion_matrix(
    model: Pipeline, features: pd.DataFrame, target: pd.Series, threshold: float | None = None
) -> Figure:
    threshold = CONFIG.training.decision_threshold if threshold is None else threshold
    predictions = (model.predict_proba(features)[:, 1] >= threshold).astype(int)

    figure, axis = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        target, predictions, display_labels=["rejected", "approved"], colorbar=False, ax=axis
    )
    axis.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_threshold_cost(
    target: pd.Series,
    probabilities: np.ndarray,
    false_approval_cost: float | None = None,
    false_rejection_cost: float | None = None,
) -> Figure:
    costs = threshold_cost_curve(target, probabilities, false_approval_cost, false_rejection_cost)
    best = optimal_threshold(target, probabilities, false_approval_cost, false_rejection_cost)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(costs["threshold"], costs["expected_cost"], label="expected cost")
    axis.axvline(best, ls="--", c="#c44e52", label=f"cost-optimal ({best:.2f})")
    axis.axvline(0.5, ls=":", c="grey", label="default (0.50)")
    axis.set(xlabel="decision threshold", ylabel="expected cost", title="Cost of each threshold")
    axis.legend()
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_learning_curve(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str = DEFAULT_MODEL_NAME,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> Figure:
    """Does the model need more data, or is it capacity-limited?"""
    model = build_model(model_name, numeric_features, categorical_features, params)
    sizes, train_scores, validation_scores = learning_curve(
        model,
        features,
        target,
        cv=cross_validation_folds(),
        scoring=SCORING,
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1,
        random_state=CONFIG.seed,
    )

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for scores, label, color in [
        (train_scores, "training", "#4c72b0"),
        (validation_scores, "cross-validation", "#c44e52"),
    ]:
        mean, deviation = scores.mean(axis=1), scores.std(axis=1)
        axis.plot(sizes, mean, marker="o", color=color, label=label)
        axis.fill_between(sizes, mean - deviation, mean + deviation, alpha=0.15, color=color)

    axis.set(xlabel="training rows", ylabel="PR-AUC", title=f"Learning curve ({model_name})")
    axis.legend()
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_model_comparison(scores: pd.DataFrame) -> Figure:
    """Cross-validated PR-AUC per model, with the fold-to-fold spread as error bars."""
    figure, axis = plt.subplots(figsize=(7, 4))
    ordered = scores.sort_values("mean")
    axis.barh(ordered.index, ordered["mean"], xerr=ordered["std"], color="#4c72b0", alpha=0.85)
    axis.set(xlabel="cross-validated PR-AUC", title="Model comparison")
    for position, value in enumerate(ordered["mean"]):
        axis.text(value + 0.005, position, f"{value:.3f}", va="center")
    figure.tight_layout()
    plt.close(figure)
    return figure


def out_of_fold_probabilities(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str = DEFAULT_MODEL_NAME,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Predictions for rows the model did not see while fitting.

    In-sample probabilities are overconfident, so a threshold chosen on them lands far too
    high. Choosing it on out-of-fold predictions keeps the holdout untouched and honest.
    """
    model = build_model(model_name, numeric_features, categorical_features, params)
    return cross_val_predict(
        model,
        features,
        target,
        cv=cross_validation_folds(),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
