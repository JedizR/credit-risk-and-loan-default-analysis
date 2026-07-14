import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.config import CONFIG
from credit_risk.data import FEATURE_COLUMNS, TARGET_COLUMN
from credit_risk.train import (
    load_model,
    predict_applicants,
    save_metrics,
    save_model,
    split_holdout,
    train_model,
)

EXPECTED_METRICS = {
    "average_precision",
    "roc_auc",
    "brier_score",
    "threshold",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
}


def test_split_holdout_preserves_class_balance(sample_frame: pd.DataFrame) -> None:
    split = split_holdout(sample_frame, FEATURE_COLUMNS)

    full_approval_rate = sample_frame[TARGET_COLUMN].mean()
    holdout_approval_rate = split.holdout_target.mean()

    assert len(split.holdout_target) == round(len(sample_frame) * CONFIG.training.holdout_fraction)
    assert holdout_approval_rate == pytest.approx(full_approval_rate, abs=0.02)


def test_train_model_reports_every_metric(sample_frame: pd.DataFrame) -> None:
    _, metrics = train_model(sample_frame)

    assert set(metrics) == EXPECTED_METRICS


def test_train_model_beats_random_ranking(sample_frame: pd.DataFrame) -> None:
    _, metrics = train_model(sample_frame)

    assert metrics["roc_auc"] > 0.7


def test_saved_model_predicts_identically_after_reload(
    sample_frame: pd.DataFrame, tmp_path: Path
) -> None:
    model, _ = train_model(sample_frame)
    model_path = tmp_path / "nested" / "model.joblib"
    save_model(model, model_path)

    features = sample_frame[FEATURE_COLUMNS]
    reloaded_predictions = predict_applicants(load_model(model_path), features)
    original_predictions = predict_applicants(model, features)

    pd.testing.assert_frame_equal(reloaded_predictions, original_predictions)


def test_predict_applicants_appends_probability_and_decision(sample_frame: pd.DataFrame) -> None:
    model, _ = train_model(sample_frame)

    scored = predict_applicants(model, sample_frame[FEATURE_COLUMNS])

    assert scored["ApprovalProbability"].between(0.0, 1.0).all()
    assert scored["LoanApprovedPrediction"].isin([0, 1]).all()


def test_save_metrics_writes_readable_json(tmp_path: Path) -> None:
    metrics_path = tmp_path / "reports" / "metrics.json"

    save_metrics({"roc_auc": 0.91}, metrics_path)

    assert json.loads(metrics_path.read_text()) == {"roc_auc": 0.91}
