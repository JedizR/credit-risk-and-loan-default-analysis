import json
from pathlib import Path
from typing import Any

import fire

from credit_risk import silence_library_warnings
from credit_risk.config import CONFIG
from credit_risk.data import (
    PROCESSED_PARQUET,
    REGISTRY_JSON,
    load_features_to_score,
    load_training_data,
    prepare_dataset,
)
from credit_risk.explain import decision_reasons
from credit_risk.features.engineering import engineer_features
from credit_risk.pipeline import DEFAULT_MODEL_NAME
from credit_risk.train import (
    load_model,
    predict_applicants,
    save_metrics,
    save_model,
    score_model,
    split_holdout,
)
from credit_risk.versioning import build_manifest, save_run
from credit_risk.workflow import (
    TrainingOptions,
    TrainingOutcome,
    run_training,
    write_preprocessed_dataset,
)

DEFAULT_MODEL_PATH = CONFIG.paths.model
DEFAULT_METRICS_PATH = CONFIG.paths.metrics
DEFAULT_PREDICTIONS_PATH = CONFIG.paths.predictions
DEFAULT_REASONS_PATH = Path("out/decisions.csv")
PIPELINE_STAGES = ("prepare", "preprocess", "train", "evaluate")


def _path(value: Any) -> Path | None:
    """Normalise a CLI value (Fire passes strings) into a Path, keeping None as None."""
    return Path(value) if value is not None else None


def _report(metrics: dict) -> None:
    print(json.dumps(metrics, indent=2))


class CreditRisk:
    """Prepare, train, evaluate, apply and explain the loan-approval model."""

    def prepare(self) -> None:
        """Repair and convert the raw CSV to a typed parquet dataset."""
        frame = prepare_dataset()
        print(f"Prepared {len(frame)} applicants written to {PROCESSED_PARQUET}")
        print(f"Provenance written to {REGISTRY_JSON}")

    def preprocess(self, output_path: Any = None, registry_path: Any = None) -> None:
        """Engineer the features and save the model-ready dataset.

        Args:
            output_path: Where to write the model-ready parquet (default from config).
            registry_path: Where to update the provenance registry (default from config).
        """
        engineered = write_preprocessed_dataset(
            path=_path(output_path), registry_path=_path(registry_path)
        )
        output = _path(output_path) or CONFIG.paths.preprocessed_parquet
        registry = _path(registry_path) or REGISTRY_JSON
        print(f"Preprocessed {len(engineered)} applicants into {engineered.shape[1]} columns")
        print(f"Model-ready dataset written to {output}")
        print(
            "Numeric nulls are kept on purpose: the imputer is fitted per fold inside the model "
            "pipeline, so baking values in here would leak the holdout into training."
        )
        print(f"Provenance updated in {registry}")

    def train(
        self,
        data: Any = None,
        model_name: str = DEFAULT_MODEL_NAME,
        model_path: Any = DEFAULT_MODEL_PATH,
        metrics_path: Any = DEFAULT_METRICS_PATH,
        figures_path: Any = None,
        tune: bool = False,
        trials: int | None = None,
        select_features: bool = False,
        keep_outliers: bool = False,
        plots: bool = False,
    ) -> None:
        """Run the full training pipeline and record a versioned run beside the model.

        Args:
            data: Training dataset path; defaults to the prepared dataset.
            model_name: Registered model to train (lightgbm, logistic_regression, ...).
            model_path: Where to save the fitted model; version artefacts land beside it.
            metrics_path: Where to write the holdout metrics JSON.
            figures_path: Where to write figures when ``plots`` is set.
            tune: Search hyperparameters with Optuna.
            trials: Optuna trials when tuning.
            select_features: Keep only the consensus feature set.
            keep_outliers: Keep consensus outliers (removal is on by default).
            plots: Write every figure to the reports directory.
        """
        model_path = _path(model_path)
        metrics_path = _path(metrics_path)
        frame = load_training_data(_path(data))
        options = TrainingOptions(
            tune=tune,
            select_features=select_features,
            remove_outliers=not keep_outliers,
            write_plots=plots,
            trials=trials,
            figures_dir=_path(figures_path),
        )
        outcome = run_training(frame, model_name, options)

        save_model(outcome.model, model_path)
        save_metrics(outcome.metrics, metrics_path)
        run_dir = model_path.parent
        record = save_run(
            outcome.model,
            self._manifest(outcome, options, model_name),
            models_dir=run_dir,
            cards_dir=run_dir / "model_cards",
            registry_path=run_dir / "registry.json",
        )

        print(f"Trained {model_name} on {len(frame)} applicants")
        if outcome.outliers_removed:
            print(f"Removed {outcome.outliers_removed} outliers from the training rows")
        print(f"Using {len(outcome.features)} features: {', '.join(outcome.features)}")
        if outcome.params:
            print(f"Tuned parameters: {json.dumps(outcome.params)}")
        print(f"Decision threshold: {outcome.threshold}")
        print(f"Model written to {model_path}")
        print(f"Metrics written to {metrics_path}")
        if outcome.figures:
            print(f"Wrote {len(outcome.figures)} figures to {outcome.figures[0].parent}")
        print(f"Recorded run {record.run_id}; model card at {record.card_path}")
        _report(outcome.metrics)

    def run(
        self,
        stage: str = "all",
        data: Any = None,
        model_name: str = DEFAULT_MODEL_NAME,
        model_path: Any = DEFAULT_MODEL_PATH,
        metrics_path: Any = DEFAULT_METRICS_PATH,
        figures_path: Any = None,
        tune: bool = False,
        trials: int | None = None,
        select_features: bool = False,
        keep_outliers: bool = False,
        plots: bool = False,
        output_path: Any = None,
        registry_path: Any = None,
        threshold: float | None = None,
    ) -> None:
        """Run the whole pipeline, or a single stage.

        Args:
            stage: One of all, prepare, preprocess, train, evaluate. 'all' chains them in order.

        The remaining arguments are forwarded to the matching stage (for example ``--tune`` and
        ``--plots`` to train, ``--threshold`` to evaluate).
        """
        stage_runners = {
            "prepare": lambda: self.prepare(),
            "preprocess": lambda: self.preprocess(output_path, registry_path),
            "train": lambda: self.train(
                data,
                model_name,
                model_path,
                metrics_path,
                figures_path,
                tune,
                trials,
                select_features,
                keep_outliers,
                plots,
            ),
            "evaluate": lambda: self.evaluate(data, model_path, threshold),
        }
        if stage != "all" and stage not in stage_runners:
            raise ValueError(
                f"Unknown stage '{stage}'. Choose from: all, {', '.join(PIPELINE_STAGES)}"
            )
        for name in PIPELINE_STAGES if stage == "all" else (stage,):
            print(f"=== {name} ===")
            stage_runners[name]()

    def evaluate(
        self, data: Any = None, model_path: Any = DEFAULT_MODEL_PATH, threshold: float | None = None
    ) -> None:
        """Score a saved model on the held-out applicants.

        Args:
            data: Dataset to split for the holdout; defaults to the prepared dataset.
            model_path: The saved model to evaluate.
            threshold: Decision threshold; defaults to the configured value.
        """
        model = load_model(_path(model_path))
        frame = engineer_features(load_training_data(_path(data)))
        split = split_holdout(frame, list(model.feature_names_in_))
        metrics = score_model(model, split.holdout_features, split.holdout_target, threshold)
        print(f"Evaluated {_path(model_path)} on {len(split.holdout_target)} held-out applicants")
        _report(metrics)

    def predict(
        self,
        input_path: Any,
        output_path: Any = DEFAULT_PREDICTIONS_PATH,
        model_path: Any = DEFAULT_MODEL_PATH,
        threshold: float | None = None,
    ) -> None:
        """Score new applicants from a CSV.

        Args:
            input_path: CSV of applicants to score.
            output_path: Where to write the predictions CSV.
            model_path: The saved model to score with.
            threshold: Decision threshold; defaults to the configured value.
        """
        model = load_model(_path(model_path))
        features = engineer_features(load_features_to_score(_path(input_path)))[
            list(model.feature_names_in_)
        ]
        scored = predict_applicants(model, features, threshold)
        output_path = _path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)
        approved = int(scored["LoanApprovedPrediction"].sum())
        print(f"Scored {len(scored)} applicants, {approved} predicted approved")
        print(f"Predictions written to {output_path}")

    def explain(
        self,
        input_path: Any,
        output_path: Any = DEFAULT_REASONS_PATH,
        model_path: Any = DEFAULT_MODEL_PATH,
        threshold: float | None = None,
        limit: int = 100,
    ) -> None:
        """Score applicants with a human-readable reason for each decision.

        Args:
            input_path: CSV of applicants to explain.
            output_path: Where to write the decisions CSV.
            model_path: The saved model to use.
            threshold: Decision threshold; defaults to the configured value.
            limit: Number of applicants to explain.
        """
        model = load_model(_path(model_path))
        features = engineer_features(load_features_to_score(_path(input_path)))[
            list(model.feature_names_in_)
        ].head(limit)
        decisions = decision_reasons(model, features, threshold)
        output_path = _path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        decisions.to_csv(output_path, index=False)
        print(f"Explained {len(decisions)} applicants")
        print(f"Decisions and reasons written to {output_path}")
        print(decisions.head(5).to_string())

    @staticmethod
    def _manifest(outcome: TrainingOutcome, options: TrainingOptions, model_name: str) -> dict:
        return build_manifest(
            model_name=model_name,
            params=outcome.params,
            features=outcome.features,
            threshold=outcome.threshold,
            metrics=outcome.metrics,
            options={
                "tune": options.tune,
                "select_features": options.select_features,
                "remove_outliers": options.remove_outliers,
            },
        )


def main() -> None:
    """Silence library warnings, then dispatch the CLI with Fire."""
    silence_library_warnings()
    fire.Fire(CreditRisk, name="credit-risk")
