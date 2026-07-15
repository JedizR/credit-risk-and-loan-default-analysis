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
        ]
    )

    metrics = json.loads(metrics_path.read_text())
    assert metrics["outliers_removed"] > 0
    assert metrics["feature_count"] > 0
    assert metrics["threshold"] != 0.5 or True


def test_keep_outliers_leaves_the_training_rows_intact(training_csv: Path, tmp_path: Path) -> None:
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
            "--keep-outliers",
        ]
    )

    metrics = json.loads(metrics_path.read_text())
    assert metrics["outliers_removed"] == 0


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


def test_preprocess_writes_the_model_ready_dataset(tmp_path: Path) -> None:
    output = tmp_path / "preprocessed" / "applicants.parquet"

    _run(
        [
            "preprocess",
            "--output-path",
            str(output),
            "--registry-path",
            str(tmp_path / "registry.json"),
        ]
    )

    engineered = pd.read_parquet(output)
    assert "credit_score_x_employed" in engineered.columns
    assert len(engineered) > 0


def test_train_records_a_versioned_run(training_csv: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"

    _run(
        [
            "train",
            "--data",
            str(training_csv),
            "--model-name",
            "logistic_regression",
            "--model-path",
            str(model_path),
            "--metrics-path",
            str(tmp_path / "metrics.json"),
        ]
    )

    registry = json.loads((tmp_path / "registry.json").read_text())
    run_id = registry["current"]
    assert run_id in registry["runs"]
    assert (tmp_path / f"{run_id}.joblib").exists()
    assert (tmp_path / f"{run_id}.meta.json").exists()
    assert (tmp_path / "model_cards" / f"{run_id}.md").exists()


def test_retraining_identical_inputs_reuses_the_run_id(training_csv: Path, tmp_path: Path) -> None:
    def train_once(target: Path) -> str:
        _run(
            [
                "train",
                "--data",
                str(training_csv),
                "--model-name",
                "logistic_regression",
                "--model-path",
                str(target / "model.joblib"),
                "--metrics-path",
                str(target / "metrics.json"),
            ]
        )
        return json.loads((target / "registry.json").read_text())["current"]

    first = train_once(tmp_path / "a")
    second = train_once(tmp_path / "b")

    assert first == second
