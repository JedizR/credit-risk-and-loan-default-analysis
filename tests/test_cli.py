import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.cli import CreditRisk
from credit_risk.data import io
from credit_risk.models import UnknownModelError

cli = CreditRisk()


def test_prepare_writes_parquet_and_registry(
    training_csv: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(io, "RAW_CSV", training_csv)
    monkeypatch.setattr(io, "PROCESSED_PARQUET", tmp_path / "processed" / "applicants.parquet")
    monkeypatch.setattr(io, "REGISTRY_JSON", tmp_path / "registry.json")

    cli.prepare()

    assert (tmp_path / "processed" / "applicants.parquet").exists()
    assert (tmp_path / "registry.json").exists()


def test_train_writes_model_and_metrics(training_csv: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "models" / "model.joblib"
    metrics_path = tmp_path / "reports" / "metrics.json"

    cli.train(data=str(training_csv), model_path=str(model_path), metrics_path=str(metrics_path))

    assert model_path.exists()
    assert "roc_auc" in json.loads(metrics_path.read_text())


def test_predict_writes_a_score_for_every_applicant(
    training_csv: Path, scoring_csv: Path, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.joblib"
    predictions_path = tmp_path / "predictions.csv"
    cli.train(data=str(training_csv), model_path=str(model_path))

    cli.predict(
        input_path=str(scoring_csv), model_path=str(model_path), output_path=str(predictions_path)
    )

    predictions = pd.read_csv(predictions_path)
    assert len(predictions) == len(pd.read_csv(scoring_csv))
    assert {"ApprovalProbability", "LoanApprovedPrediction"}.issubset(predictions.columns)


def test_evaluate_reports_metrics_for_a_saved_model(
    training_csv: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path = tmp_path / "model.joblib"
    cli.train(data=str(training_csv), model_path=str(model_path))
    capsys.readouterr()

    cli.evaluate(data=str(training_csv), model_path=str(model_path))

    assert "roc_auc" in capsys.readouterr().out


def test_train_rejects_an_unregistered_model(training_csv: Path, tmp_path: Path) -> None:
    with pytest.raises(UnknownModelError, match="neural_net"):
        cli.train(
            data=str(training_csv),
            model_name="neural_net",
            model_path=str(tmp_path / "model.joblib"),
            metrics_path=str(tmp_path / "metrics.json"),
        )


def test_train_writes_figures_when_plots_requested(training_csv: Path, tmp_path: Path) -> None:
    figures = tmp_path / "figures"

    cli.train(
        data=str(training_csv),
        model_name="lightgbm",
        model_path=str(tmp_path / "model.joblib"),
        metrics_path=str(tmp_path / "metrics.json"),
        figures_path=str(figures),
        plots=True,
    )

    written = sorted(path.name for path in figures.glob("*.png"))
    assert any(name.startswith("dynamics_") for name in written)
    assert any(name.startswith("explain_") for name in written)
    assert any(name.startswith("errors_") for name in written)


def test_train_records_selection_and_outlier_removal(training_csv: Path, tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"

    cli.train(
        data=str(training_csv),
        model_name="logistic_regression",
        model_path=str(tmp_path / "model.joblib"),
        metrics_path=str(metrics_path),
        select_features=True,
    )

    metrics = json.loads(metrics_path.read_text())
    assert metrics["outliers_removed"] > 0
    assert metrics["feature_count"] > 0


def test_keep_outliers_leaves_the_training_rows_intact(training_csv: Path, tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"

    cli.train(
        data=str(training_csv),
        model_name="logistic_regression",
        model_path=str(tmp_path / "model.joblib"),
        metrics_path=str(metrics_path),
        keep_outliers=True,
    )

    assert json.loads(metrics_path.read_text())["outliers_removed"] == 0


def test_run_train_stage_writes_model_and_metrics(training_csv: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "models" / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    cli.run(
        stage="train",
        data=str(training_csv),
        model_name="logistic_regression",
        model_path=str(model_path),
        metrics_path=str(metrics_path),
    )

    assert model_path.exists()
    assert "roc_auc" in json.loads(metrics_path.read_text())


def test_run_all_chains_every_stage(
    training_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(io, "RAW_CSV", training_csv)
    monkeypatch.setattr(io, "PROCESSED_PARQUET", tmp_path / "processed.parquet")
    monkeypatch.setattr(io, "REGISTRY_JSON", tmp_path / "data_registry.json")

    cli.run(
        stage="all",
        model_name="logistic_regression",
        data=str(tmp_path / "processed.parquet"),
        model_path=str(tmp_path / "models" / "model.joblib"),
        metrics_path=str(tmp_path / "metrics.json"),
        output_path=str(tmp_path / "preprocessed.parquet"),
        registry_path=str(tmp_path / "data_registry.json"),
    )

    out = capsys.readouterr().out
    for stage in ("prepare", "preprocess", "train", "evaluate"):
        assert f"=== {stage} ===" in out
    assert (tmp_path / "processed.parquet").exists()
    assert (tmp_path / "preprocessed.parquet").exists()
    assert (tmp_path / "models" / "model.joblib").exists()


def test_run_rejects_an_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown stage"):
        cli.run(stage="frobnicate")


def test_explain_emits_a_reason_for_every_applicant(
    training_csv: Path, scoring_csv: Path, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.joblib"
    decisions_path = tmp_path / "decisions.csv"
    cli.train(
        data=str(training_csv),
        model_name="lightgbm",
        model_path=str(model_path),
        metrics_path=str(tmp_path / "metrics.json"),
    )

    cli.explain(
        input_path=str(scoring_csv),
        model_path=str(model_path),
        output_path=str(decisions_path),
        limit=25,
    )

    decisions = pd.read_csv(decisions_path)
    assert len(decisions) == 25
    assert set(decisions.columns) == {"decision", "probability", "reasons"}
    assert decisions["reasons"].str.len().gt(0).all()


def test_preprocess_writes_the_model_ready_dataset(tmp_path: Path) -> None:
    output = tmp_path / "preprocessed" / "applicants.parquet"

    cli.preprocess(output_path=str(output), registry_path=str(tmp_path / "registry.json"))

    engineered = pd.read_parquet(output)
    assert "credit_score_x_employed" in engineered.columns
    assert len(engineered) > 0


def test_train_records_a_versioned_run(training_csv: Path, tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"

    cli.train(
        data=str(training_csv),
        model_name="logistic_regression",
        model_path=str(model_path),
        metrics_path=str(tmp_path / "metrics.json"),
    )

    registry = json.loads((tmp_path / "registry.json").read_text())
    run_id = registry["current"]
    assert run_id in registry["runs"]
    assert (tmp_path / f"{run_id}.joblib").exists()
    assert (tmp_path / f"{run_id}.meta.json").exists()
    assert (tmp_path / "model_cards" / f"{run_id}.md").exists()


def test_retraining_identical_inputs_reuses_the_run_id(training_csv: Path, tmp_path: Path) -> None:
    def train_once(target: Path) -> str:
        cli.train(
            data=str(training_csv),
            model_name="logistic_regression",
            model_path=str(target / "model.joblib"),
            metrics_path=str(target / "metrics.json"),
        )
        return json.loads((target / "registry.json").read_text())["current"]

    assert train_once(tmp_path / "a") == train_once(tmp_path / "b")
