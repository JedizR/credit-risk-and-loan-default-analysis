import json

import pandas as pd

from credit_risk.config import CONFIG
from credit_risk.data.schema import TARGET_COLUMN
from credit_risk.workflow import TrainingOptions, TrainingOutcome, run_training


def test_default_run_trains_and_scores(sample_frame: pd.DataFrame) -> None:
    outcome = run_training(sample_frame, "logistic_regression")

    assert isinstance(outcome, TrainingOutcome)
    assert 0.0 <= outcome.metrics["average_precision"] <= 1.0
    assert outcome.metrics["feature_count"] == len(outcome.features)
    assert outcome.outliers_removed > 0
    assert outcome.figures == []


def test_threshold_is_chosen_not_defaulted(sample_frame: pd.DataFrame) -> None:
    outcome = run_training(sample_frame, "logistic_regression")

    assert 0.05 <= outcome.threshold <= 0.95
    assert outcome.metrics["threshold"] == outcome.threshold


def test_sensitive_features_never_enter_the_model(sample_frame: pd.DataFrame) -> None:
    outcome = run_training(sample_frame, "logistic_regression")

    for sensitive in CONFIG.sensitive_features:
        assert sensitive not in outcome.features


def test_keeping_outliers_leaves_all_rows(sample_frame: pd.DataFrame) -> None:
    outcome = run_training(
        sample_frame, "logistic_regression", TrainingOptions(remove_outliers=False)
    )

    assert outcome.outliers_removed == 0
    assert outcome.metrics["outliers_removed"] == 0


def test_selection_trims_the_feature_set(sample_frame: pd.DataFrame) -> None:
    baseline = run_training(sample_frame, "logistic_regression")
    selected = run_training(
        sample_frame, "logistic_regression", TrainingOptions(select_features=True)
    )

    assert len(selected.features) <= len(baseline.features)


def test_tuning_records_the_chosen_params(sample_frame: pd.DataFrame) -> None:
    outcome = run_training(
        sample_frame, "logistic_regression", TrainingOptions(tune=True, trials=3)
    )

    assert "C" in outcome.params


def test_plots_are_written_to_the_reports_directory(sample_frame: pd.DataFrame, tmp_path) -> None:
    outcome = run_training(
        sample_frame,
        "lightgbm",
        TrainingOptions(write_plots=True, figures_dir=tmp_path / "figures"),
    )

    assert len(outcome.figures) >= 10
    assert all(path.exists() for path in outcome.figures)
    assert any("explain_beeswarm" in path.name for path in outcome.figures)
    assert any("errors_" in path.name for path in outcome.figures)


def test_preprocessed_dataset_is_the_model_ready_frame(
    sample_frame: pd.DataFrame, tmp_path
) -> None:
    from credit_risk.features.engineering import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC
    from credit_risk.workflow import write_preprocessed_dataset

    path = tmp_path / "preprocessed" / "applicants.parquet"

    engineered = write_preprocessed_dataset(sample_frame, path, tmp_path / "registry.json")

    assert path.exists()
    restored = pd.read_parquet(path)
    assert set(ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL).issubset(restored.columns)
    assert TARGET_COLUMN in restored.columns
    assert len(restored) == len(sample_frame)
    assert restored.dtypes.to_dict() == engineered.dtypes.to_dict()


def test_preprocessing_adds_to_provenance_without_dropping_it(
    sample_frame: pd.DataFrame, tmp_path
) -> None:
    from credit_risk.workflow import write_preprocessed_dataset

    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"raw_sha256": "kept", "processed_sha256": "kept"}))

    write_preprocessed_dataset(sample_frame, tmp_path / "applicants.parquet", registry)

    after = json.loads(registry.read_text())
    # The earlier stages must survive: the record is merged, not overwritten.
    assert after["raw_sha256"] == "kept"
    assert after["processed_sha256"] == "kept"
    assert len(after["preprocessed_sha256"]) == 64
    assert after["preprocessed_rows"] == len(sample_frame)


def test_preprocessing_never_touches_the_project_registry(
    sample_frame: pd.DataFrame, tmp_path
) -> None:
    from credit_risk.workflow import write_preprocessed_dataset

    project_registry = CONFIG.paths.registry_json
    before = project_registry.read_text() if project_registry.exists() else None

    write_preprocessed_dataset(
        sample_frame, tmp_path / "applicants.parquet", tmp_path / "registry.json"
    )

    after = project_registry.read_text() if project_registry.exists() else None
    assert after == before
