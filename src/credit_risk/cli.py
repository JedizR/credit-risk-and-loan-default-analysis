import argparse
import json
from pathlib import Path

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
from credit_risk.pipeline import DEFAULT_MODEL_NAME, MODEL_BUILDERS
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


def _report(metrics: dict[str, float]) -> None:
    print(json.dumps(metrics, indent=2))


def run_prepare(_args: argparse.Namespace) -> None:
    frame = prepare_dataset()
    print(f"Prepared {len(frame)} applicants written to {PROCESSED_PARQUET}")
    print(f"Provenance written to {REGISTRY_JSON}")


def run_preprocess(args: argparse.Namespace) -> None:
    engineered = write_preprocessed_dataset(path=args.output_path, registry_path=args.registry_path)
    output = args.output_path or CONFIG.paths.preprocessed_parquet
    registry = args.registry_path or REGISTRY_JSON

    print(f"Preprocessed {len(engineered)} applicants into {engineered.shape[1]} columns")
    print(f"Model-ready dataset written to {output}")
    print(
        "Numeric nulls are kept on purpose: the imputer is fitted per fold inside the model "
        "pipeline, so baking values in here would leak the holdout into training."
    )
    print(f"Provenance updated in {registry}")


def run_train(args: argparse.Namespace) -> None:
    frame = load_training_data(args.data)
    options = TrainingOptions(
        tune=args.tune,
        select_features=args.select_features,
        remove_outliers=not args.keep_outliers,
        write_plots=args.plots,
        trials=args.trials,
        figures_dir=args.figures_path,
    )
    outcome = run_training(frame, args.model_name, options)

    save_model(outcome.model, args.model_path)
    save_metrics(outcome.metrics, args.metrics_path)
    # Version artefacts live next to the working model, so a custom --model-path keeps them close.
    run_dir = args.model_path.parent
    record = save_run(
        outcome.model,
        _manifest_for(outcome, options, args.model_name),
        models_dir=run_dir,
        cards_dir=run_dir / "model_cards",
        registry_path=run_dir / "registry.json",
    )

    print(f"Trained {args.model_name} on {len(frame)} applicants")
    if outcome.outliers_removed:
        print(f"Removed {outcome.outliers_removed} outliers from the training rows")
    print(f"Using {len(outcome.features)} features: {', '.join(outcome.features)}")
    if outcome.params:
        print(f"Tuned parameters: {json.dumps(outcome.params)}")
    print(f"Decision threshold: {outcome.threshold}")
    print(f"Model written to {args.model_path}")
    print(f"Metrics written to {args.metrics_path}")
    if outcome.figures:
        print(f"Wrote {len(outcome.figures)} figures to {outcome.figures[0].parent}")
    print(f"Recorded run {record.run_id}; model card at {record.card_path}")
    _report(outcome.metrics)


def _manifest_for(outcome: TrainingOutcome, options: TrainingOptions, model_name: str) -> dict:
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


def run_evaluate(args: argparse.Namespace) -> None:
    frame = engineer_features(load_training_data(args.data))
    model = load_model(args.model_path)
    features = list(model.feature_names_in_)
    split = split_holdout(frame, features)

    metrics = score_model(model, split.holdout_features, split.holdout_target, args.threshold)

    print(f"Evaluated {args.model_path} on {len(split.holdout_target)} held-out applicants")
    _report(metrics)


def run_predict(args: argparse.Namespace) -> None:
    model = load_model(args.model_path)
    features = engineer_features(load_features_to_score(args.input_path))[
        list(model.feature_names_in_)
    ]

    scored = predict_applicants(model, features, args.threshold)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_path, index=False)

    approved = int(scored["LoanApprovedPrediction"].sum())
    print(f"Scored {len(scored)} applicants, {approved} predicted approved")
    print(f"Predictions written to {args.output_path}")


def run_explain(args: argparse.Namespace) -> None:
    model = load_model(args.model_path)
    features = engineer_features(load_features_to_score(args.input_path))[
        list(model.feature_names_in_)
    ].head(args.limit)

    decisions = decision_reasons(model, features, args.threshold)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(args.output_path, index=False)

    print(f"Explained {len(decisions)} applicants")
    print(f"Decisions and reasons written to {args.output_path}")
    print(decisions.head(5).to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit-risk",
        description="Prepare, train, evaluate, apply and explain the loan approval model.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser(
        "prepare", help="Repair and convert the raw CSV to a typed parquet dataset"
    )
    prepare.set_defaults(handler=run_prepare)

    preprocess = subcommands.add_parser(
        "preprocess", help="Engineer the features and save the model-ready dataset"
    )
    preprocess.add_argument("--output-path", type=Path, default=None)
    preprocess.add_argument("--registry-path", type=Path, default=None)
    preprocess.set_defaults(handler=run_preprocess)

    train = subcommands.add_parser("train", help="Run the full training pipeline")
    train.add_argument("--data", type=Path, default=None, help="defaults to the prepared dataset")
    train.add_argument("--model-name", choices=sorted(MODEL_BUILDERS), default=DEFAULT_MODEL_NAME)
    train.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    train.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    train.add_argument("--figures-path", type=Path, default=None)
    train.add_argument("--tune", action="store_true", help="search hyperparameters with Optuna")
    train.add_argument("--trials", type=int, default=None, help="Optuna trials when tuning")
    train.add_argument(
        "--select-features", action="store_true", help="keep only the consensus feature set"
    )
    train.add_argument(
        "--keep-outliers", action="store_true", help="keep consensus outliers in the training rows"
    )
    train.add_argument(
        "--plots", action="store_true", help="write every figure to the reports directory"
    )
    train.set_defaults(handler=run_train)

    evaluate = subcommands.add_parser(
        "evaluate", help="Score a saved model on the held-out applicants"
    )
    evaluate.add_argument("--data", type=Path, default=None)
    evaluate.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    evaluate.add_argument("--threshold", type=float, default=None)
    evaluate.set_defaults(handler=run_evaluate)

    predict = subcommands.add_parser("predict", help="Score new applicants from a CSV")
    predict.add_argument("--input-path", type=Path, required=True)
    predict.add_argument("--output-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    predict.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    predict.add_argument("--threshold", type=float, default=None)
    predict.set_defaults(handler=run_predict)

    explain = subcommands.add_parser(
        "explain", help="Score applicants with a human-readable reason for each decision"
    )
    explain.add_argument("--input-path", type=Path, required=True)
    explain.add_argument("--output-path", type=Path, default=DEFAULT_REASONS_PATH)
    explain.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    explain.add_argument("--threshold", type=float, default=None)
    explain.add_argument("--limit", type=int, default=100, help="applicants to explain")
    explain.set_defaults(handler=run_explain)

    return parser


def main() -> None:
    silence_library_warnings()
    args = build_parser().parse_args()
    args.handler(args)
