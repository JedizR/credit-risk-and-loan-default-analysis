import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.cli import build_parser, main


def _run(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


def test_train_writes_model_and_metrics(training_csv: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "models" / "model.joblib"
    metrics_path = tmp_path / "reports" / "metrics.json"

    _run(
        [
            "train",
            "--data",
            str(training_csv),
            "--model-path",
            str(model_path),
            "--metrics-path",
            str(metrics_path),
        ]
    )

    assert model_path.exists()
    assert "roc_auc" in json.loads(metrics_path.read_text())


def test_predict_writes_a_score_for_every_applicant(
    training_csv: Path, scoring_csv: Path, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.joblib"
    predictions_path = tmp_path / "predictions.csv"
    _run(["train", "--data", str(training_csv), "--model-path", str(model_path)])

    _run(
        [
            "predict",
            "--input-path",
            str(scoring_csv),
            "--model-path",
            str(model_path),
            "--output-path",
            str(predictions_path),
        ]
    )

    predictions = pd.read_csv(predictions_path)
    assert len(predictions) == len(pd.read_csv(scoring_csv))
    assert {"ApprovalProbability", "LoanApprovedPrediction"}.issubset(predictions.columns)


def test_evaluate_reports_metrics_for_a_saved_model(
    training_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path = tmp_path / "model.joblib"
    _run(["train", "--data", str(training_csv), "--model-path", str(model_path)])
    capsys.readouterr()

    _run(["evaluate", "--data", str(training_csv), "--model-path", str(model_path)])

    assert "roc_auc" in capsys.readouterr().out


def test_train_rejects_an_unregistered_model(training_csv: Path) -> None:
    with pytest.raises(SystemExit):
        _run(["train", "--data", str(training_csv), "--model-name", "neural_net"])


def test_main_requires_a_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["credit-risk"])

    with pytest.raises(SystemExit):
        main()
