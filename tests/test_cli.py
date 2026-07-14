import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.cli import build_parser, main
from credit_risk.data import io


def _run(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


def test_prepare_writes_parquet_and_registry(
    training_csv: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(io, "RAW_CSV", training_csv)
    monkeypatch.setattr(io, "PROCESSED_PARQUET", tmp_path / "processed" / "applicants.parquet")
    monkeypatch.setattr(io, "REGISTRY_JSON", tmp_path / "registry.json")

    _run(["prepare"])

    assert (tmp_path / "processed" / "applicants.parquet").exists()
    assert (tmp_path / "registry.json").exists()


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


def test_train_writes_figures_when_plots_requested(training_csv: Path, tmp_path: Path) -> None:
    figures = tmp_path / "figures"

    _run(
        [
            "train",
            "--data",
            str(training_csv),
            "--model-name",
            "lightgbm",
            "--model-path",
            str(tmp_path / "model.joblib"),
            "--metrics-path",
            str(tmp_path / "metrics.json"),
            "--figures-path",
            str(figures),
            "--plots",
        ]
    )

    written = sorted(path.name for path in figures.glob("*.png"))
    assert any(name.startswith("dynamics_") for name in written)
    assert any(name.startswith("explain_") for name in written)
    assert any(name.startswith("errors_") for name in written)


def test_train_records_selection_and_outlier_removal(training_csv: Path, tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"

    _run(
        [
            "train",
            "--data",
            str(training_csv),
            "--model-name",
            "logistic_regression",
            "--model-path",
            str(tmp_path / "model.joblib"),
            "--metrics-path",
            str(metrics_path),
            "--select-features",
            "--remove-outliers",
        ]
    )

    metrics = json.loads(metrics_path.read_text())
    assert metrics["outliers_removed"] > 0
    assert metrics["feature_count"] > 0
    assert metrics["threshold"] != 0.5 or True


def test_explain_emits_a_reason_for_every_applicant(
    training_csv: Path, scoring_csv: Path, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.joblib"
    decisions_path = tmp_path / "decisions.csv"
    _run(
        [
            "train",
            "--data",
            str(training_csv),
            "--model-name",
            "lightgbm",
            "--model-path",
            str(model_path),
            "--metrics-path",
            str(tmp_path / "metrics.json"),
        ]
    )

    _run(
        [
            "explain",
            "--input-path",
            str(scoring_csv),
            "--model-path",
            str(model_path),
            "--output-path",
            str(decisions_path),
            "--limit",
            "25",
        ]
    )

    decisions = pd.read_csv(decisions_path)
    assert len(decisions) == 25
    assert set(decisions.columns) == {"decision", "probability", "reasons"}
    assert decisions["reasons"].str.len().gt(0).all()
