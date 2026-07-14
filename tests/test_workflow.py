import pandas as pd

from credit_risk.config import CONFIG
from credit_risk.workflow import TrainingOptions, TrainingOutcome, run_training


def test_default_run_trains_and_scores(sample_frame: pd.DataFrame) -> None:
    outcome = run_training(sample_frame, "logistic_regression")

    assert isinstance(outcome, TrainingOutcome)
    assert 0.0 <= outcome.metrics["average_precision"] <= 1.0
    assert outcome.metrics["feature_count"] == len(outcome.features)
    assert outcome.outliers_removed == 0
    assert outcome.figures == []


def test_threshold_is_chosen_not_defaulted(sample_frame: pd.DataFrame) -> None:
    outcome = run_training(sample_frame, "logistic_regression")

    assert 0.05 <= outcome.threshold <= 0.95
    assert outcome.metrics["threshold"] == outcome.threshold


def test_sensitive_features_never_enter_the_model(sample_frame: pd.DataFrame) -> None:
    outcome = run_training(sample_frame, "logistic_regression")

    for sensitive in CONFIG.sensitive_features:
        assert sensitive not in outcome.features


def test_outlier_removal_shrinks_the_training_rows(sample_frame: pd.DataFrame) -> None:
    outcome = run_training(
        sample_frame, "logistic_regression", TrainingOptions(remove_outliers=True)
    )

    assert outcome.outliers_removed > 0
    assert outcome.metrics["outliers_removed"] == outcome.outliers_removed


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
