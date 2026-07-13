import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from credit_risk.data import FEATURE_COLUMNS, TARGET_COLUMN
from credit_risk.pipeline import DEFAULT_MODEL_NAME, RANDOM_SEED, build_model

HOLDOUT_FRACTION = 0.2
DECISION_THRESHOLD = 0.5


@dataclass(frozen=True)
class HoldoutSplit:
    train_features: pd.DataFrame
    holdout_features: pd.DataFrame
    train_target: pd.Series
    holdout_target: pd.Series


def split_holdout(frame: pd.DataFrame) -> HoldoutSplit:
    train_features, holdout_features, train_target, holdout_target = train_test_split(
        frame[FEATURE_COLUMNS],
        frame[TARGET_COLUMN],
        test_size=HOLDOUT_FRACTION,
        random_state=RANDOM_SEED,
        stratify=frame[TARGET_COLUMN],
    )
    return HoldoutSplit(train_features, holdout_features, train_target, holdout_target)


def score_model(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    default_probabilities = model.predict_proba(features)[:, 1]
    predictions = (default_probabilities >= DECISION_THRESHOLD).astype(int)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        target, predictions
    ).ravel()

    return {
        "average_precision": float(average_precision_score(target, default_probabilities)),
        "roc_auc": float(roc_auc_score(target, default_probabilities)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(target, predictions)),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
    }


def train_model(
    frame: pd.DataFrame, model_name: str = DEFAULT_MODEL_NAME
) -> tuple[Pipeline, dict[str, float]]:
    split = split_holdout(frame)
    model = build_model(model_name).fit(split.train_features, split.train_target)
    metrics = score_model(model, split.holdout_features, split.holdout_target)
    return model, metrics


def save_model(model: Pipeline, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path) -> Pipeline:
    return joblib.load(model_path)


def save_metrics(metrics: dict[str, float], metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")


def predict_applicants(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    approval_probabilities = model.predict_proba(features)[:, 1]
    return features.assign(
        ApprovalProbability=approval_probabilities,
        LoanApprovedPrediction=(approval_probabilities >= DECISION_THRESHOLD).astype(int),
    )
