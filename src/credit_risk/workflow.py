import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from credit_risk.anomaly.handling import clean_training_frame  # noqa: E402
from credit_risk.config import CONFIG  # noqa: E402
from credit_risk.data.io import load_training_data, write_parquet  # noqa: E402
from credit_risk.data.quality import frame_hash  # noqa: E402
from credit_risk.data.schema import TARGET_COLUMN  # noqa: E402
from credit_risk.error_analysis import (  # noqa: E402
    classify_predictions,
    plot_error_overview,
    plot_error_rate_by_segment,
)
from credit_risk.evaluation import (  # noqa: E402
    optimal_threshold,
    out_of_fold_probabilities,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_roc_and_pr_curves,
    plot_threshold_cost,
)
from credit_risk.explain import (  # noqa: E402
    explain_model,
    plot_beeswarm,
    plot_dependence,
    plot_importance_bar,
    top_features,
)
from credit_risk.features.engineering import (  # noqa: E402
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    engineer_features,
)
from credit_risk.features.selection import (  # noqa: E402
    candidate_features,
    plot_selection_curve,
    select_features,
    split_feature_types,
)
from credit_risk.pipeline import DEFAULT_MODEL_NAME, build_model  # noqa: E402
from credit_risk.train import calibrate_model, score_model, split_frame  # noqa: E402
from credit_risk.tuning import plot_param_importances, plot_tuning_history, tune_model  # noqa: E402

ERROR_SEGMENTS = ["EmploymentType", "credit_band"]


@dataclass(frozen=True)
class TrainingOptions:
    """What the run should do. A value object so callers never pass bare booleans."""

    tune: bool = False
    select_features: bool = False
    remove_outliers: bool = True
    write_plots: bool = False
    trials: int | None = None
    figures_dir: Path | None = None


@dataclass(frozen=True)
class TrainingOutcome:
    model: Pipeline
    metrics: dict[str, float]
    features: list[str]
    params: dict[str, Any]
    threshold: float
    outliers_removed: int
    figures: list[Path] = field(default_factory=list)


def run_training(
    frame: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    options: TrainingOptions | None = None,
) -> TrainingOutcome:
    """Engineer, clean, select, tune, fit, threshold and score — the notebook flow, headless.

    Every learned decision (outlier removal, feature selection, tuning, threshold) is made on the
    training rows alone; the holdout is touched exactly once, to report the final numbers. The
    decision threshold is chosen on out-of-fold training predictions rather than in-sample scores
    (which are overconfident), so the holdout stays untouched until that final score.
    """
    options = options or TrainingOptions()

    engineered = engineer_features(frame)
    train_frame, holdout_frame = split_frame(engineered)

    outliers_removed = 0
    if options.remove_outliers:
        cleaned, mask = clean_training_frame(train_frame)
        outliers_removed = int(mask.sum())
        train_frame = cleaned

    features = candidate_features()
    train_target = train_frame[TARGET_COLUMN]
    if options.select_features:
        features, _ = select_features(train_frame[features], train_target, model_name=model_name)

    numeric, categorical = split_feature_types(features)
    train_features = train_frame[features]

    params: dict[str, Any] = {}
    study = None
    if options.tune:
        study = tune_model(
            train_features, train_target, model_name, numeric, categorical, options.trials
        )
        params = dict(study.best_params)

    model = build_model(model_name, numeric, categorical, params)
    model.fit(train_features, train_target)

    out_of_fold = out_of_fold_probabilities(
        train_features, train_target, model_name, numeric, categorical, params
    )
    threshold = optimal_threshold(train_target, out_of_fold)

    holdout_features = holdout_frame[features]
    holdout_target = holdout_frame[TARGET_COLUMN]
    metrics = score_model(model, holdout_features, holdout_target, threshold)

    calibrated = calibrate_model(
        build_model(model_name, numeric, categorical, params), train_features, train_target
    )
    metrics["brier_score_calibrated"] = float(
        score_model(calibrated, holdout_features, holdout_target, threshold)["brier_score"]
    )
    metrics["outliers_removed"] = outliers_removed
    metrics["feature_count"] = len(features)

    figures: list[Path] = []
    if options.write_plots:
        figures = write_figures(
            model,
            calibrated,
            train_features,
            train_target,
            holdout_features,
            holdout_target,
            holdout_frame,
            model_name,
            params,
            threshold,
            study,
            options.figures_dir,
        )

    return TrainingOutcome(model, metrics, features, params, threshold, outliers_removed, figures)


def write_figures(
    model: Pipeline,
    calibrated: Pipeline,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    holdout_features: pd.DataFrame,
    holdout_target: pd.Series,
    holdout_frame: pd.DataFrame,
    model_name: str,
    params: dict[str, Any],
    threshold: float,
    study: Any | None = None,
    directory: Path | None = None,
) -> list[Path]:
    """Write every figure the notebook produces into the reports directory.

    Feature selection may have dropped columns from the model, so two things are derived from what
    the fitted model actually uses: the error-analysis context columns (to describe *who* it fails
    on) are re-attached from the holdout frame, and the dependence-plot feature names come from the
    model's own top features rather than being hard-coded.
    """
    directory = directory or CONFIG.paths.figures
    directory.mkdir(parents=True, exist_ok=True)

    numeric, categorical = split_feature_types(list(train_features.columns))
    probabilities = model.predict_proba(holdout_features)[:, 1]
    classified = classify_predictions(model, holdout_features, holdout_target, threshold)
    for segment in ERROR_SEGMENTS:
        classified[segment] = holdout_frame[segment]
    explanation = explain_model(model, holdout_features)
    influential = top_features(explanation, 2)

    named_figures = {
        "dynamics_learning_curve": lambda: plot_learning_curve(
            train_features, train_target, model_name, numeric, categorical, params
        ),
        "dynamics_roc_pr": lambda: plot_roc_and_pr_curves(
            {model_name: model}, holdout_features, holdout_target
        ),
        "dynamics_calibration": lambda: plot_calibration_curve(
            {model_name: model, f"{model_name} (calibrated)": calibrated},
            holdout_features,
            holdout_target,
        ),
        "evaluation_confusion_matrix": lambda: plot_confusion_matrix(
            model, holdout_features, holdout_target, threshold
        ),
        "evaluation_threshold_cost": lambda: plot_threshold_cost(
            holdout_target, probabilities, chosen=threshold
        ),
        "explain_beeswarm": lambda: plot_beeswarm(explanation),
        "explain_importance": lambda: plot_importance_bar(explanation),
        "explain_dependence": lambda: plot_dependence(explanation, influential[0], influential[-1]),
        "errors_overview": lambda: plot_error_overview(classified),
        "errors_by_segment": lambda: plot_error_rate_by_segment(classified, ERROR_SEGMENTS),
    }
    if study is not None:
        named_figures["tuning_history"] = lambda: plot_tuning_history(study)
        named_figures["tuning_param_importance"] = lambda: plot_param_importances(study)

    written = []
    for name, make_figure in named_figures.items():
        figure = make_figure()
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def write_selection_figure(curve: pd.DataFrame, directory: Path | None = None) -> Path:
    directory = directory or CONFIG.paths.figures
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "features_selection_curve.png"

    figure = plot_selection_curve(curve)
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path


def write_preprocessed_dataset(
    frame: pd.DataFrame | None = None,
    path: Path | None = None,
    registry_path: Path | None = None,
) -> pd.DataFrame:
    """Persist the model-ready frame: the exact input the model pipeline is fitted on.

    Only the *stateless* half of preprocessing is stored. Imputation, scaling and encoding are
    deliberately left out: they are fitted on the training fold, so a globally fitted version
    saved here would leak the holdout into the training data. Those steps stay inside the
    sklearn pipeline, where they are refit per fold.
    """
    frame = load_training_data() if frame is None else frame
    path = path or CONFIG.paths.preprocessed_parquet

    engineered = engineer_features(frame)
    write_parquet(engineered, path)
    _record_preprocessed(engineered, path, registry_path or CONFIG.paths.registry_json)
    return engineered


def _record_preprocessed(engineered: pd.DataFrame, path: Path, registry_path: Path) -> None:
    """Add the model-ready dataset to the provenance record without dropping what is there."""
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}

    registry["preprocessed_parquet"] = Path(path).as_posix()
    registry["preprocessed_sha256"] = frame_hash(engineered)
    registry["preprocessed_rows"] = int(len(engineered))
    registry["preprocessed_columns"] = int(engineered.shape[1])
    registry["engineered_features"] = ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
