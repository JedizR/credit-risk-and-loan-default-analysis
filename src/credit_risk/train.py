import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from credit_risk.config import CONFIG
from credit_risk.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from credit_risk.pipeline import DEFAULT_MODEL_NAME, build_model


@dataclass(frozen=True)
class HoldoutSplit:
    """The train and holdout features and targets, bundled as one value."""

    train_features: pd.DataFrame
    holdout_features: pd.DataFrame
    train_target: pd.Series
    holdout_target: pd.Series


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/holdout split of whole rows, so row-level steps can act on the frame."""
    return train_test_split(
        frame,
        test_size=CONFIG.training.holdout_fraction,
        random_state=CONFIG.seed,
        stratify=frame[TARGET_COLUMN],
    )


def split_holdout(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> HoldoutSplit:
    """Split a frame and wrap the feature and target halves in a ``HoldoutSplit``."""
    train_frame, holdout_frame = split_frame(frame)
    columns = feature_columns or [column for column in frame.columns if column != TARGET_COLUMN]
    return HoldoutSplit(
        train_frame[columns],
        holdout_frame[columns],
        train_frame[TARGET_COLUMN],
        holdout_frame[TARGET_COLUMN],
    )


def score_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute every headline metric at a decision threshold.

    Args:
        model: A fitted pipeline.
        features: The features to score.
        target: The true labels aligned to ``features``.
        threshold: The decision threshold, or None for the configured default.

    Returns:
        A dict of PR-AUC, ROC-AUC, Brier, precision/recall/F1, accuracy, the threshold used, and
        the four confusion-matrix counts.
    """
    threshold = CONFIG.training.decision_threshold if threshold is None else threshold
    approval_probabilities = model.predict_proba(features)[:, 1]
    predictions = (approval_probabilities >= threshold).astype(int)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        target, predictions
    ).ravel()

    return {
        "average_precision": float(average_precision_score(target, approval_probabilities)),
        "roc_auc": float(roc_auc_score(target, approval_probabilities)),
        "brier_score": float(brier_score_loss(target, approval_probabilities)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(target, predictions)),
        "threshold": float(threshold),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
    }


def calibrate_model(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> Pipeline:
    """Turn raw classifier scores into probabilities that match observed approval rates."""
    calibrated = CalibratedClassifierCV(
        model, method="isotonic", cv=CONFIG.training.cross_validation_folds
    )
    return calibrated.fit(features, target)


def train_model(
    frame: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    numeric_features: list[str] = NUMERIC_FEATURES,
    categorical_features: list[str] = CATEGORICAL_FEATURES,
    params: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> tuple[Pipeline, dict[str, float]]:
    """Split, build, fit and score in one call — the compact train path used in tests."""
    split = split_holdout(frame, numeric_features + categorical_features)
    model = build_model(model_name, numeric_features, categorical_features, params)
    model.fit(split.train_features, split.train_target)
    metrics = score_model(model, split.holdout_features, split.holdout_target, threshold)
    return model, metrics


def save_model(model: Pipeline, model_path: Path) -> None:
    """Persist a fitted pipeline to ``model_path`` with joblib, creating the parent directory."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path) -> Pipeline:
    """Load a joblib-persisted pipeline."""
    return joblib.load(model_path)


def save_metrics(metrics: dict[str, float], metrics_path: Path) -> None:
    """Write the metrics dict to ``metrics_path`` as indented JSON."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


def predict_applicants(
    model: Pipeline, features: pd.DataFrame, threshold: float | None = None
) -> pd.DataFrame:
    """Score applicants into a copy of ``features`` with probability and decision columns."""
    threshold = CONFIG.training.decision_threshold if threshold is None else threshold
    approval_probabilities = model.predict_proba(features)[:, 1]
    return features.assign(
        ApprovalProbability=approval_probabilities,
        LoanApprovedPrediction=(approval_probabilities >= threshold).astype(int),
    )
