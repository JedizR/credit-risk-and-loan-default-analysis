import argparse
import json
from pathlib import Path

from credit_risk.data import (
    PROCESSED_PARQUET,
    REGISTRY_JSON,
    load_features_to_score,
    load_training_data,
    prepare_dataset,
)
from credit_risk.pipeline import DEFAULT_MODEL_NAME, MODEL_BUILDERS
from credit_risk.train import (
    load_model,
    predict_applicants,
    save_metrics,
    save_model,
    score_model,
    split_holdout,
    train_model,
)

DEFAULT_MODEL_PATH = Path("models/model.joblib")
DEFAULT_METRICS_PATH = Path("reports/metrics.json")
DEFAULT_PREDICTIONS_PATH = Path("out/predictions.csv")


def _report(metrics: dict[str, float]) -> None:
    print(json.dumps(metrics, indent=2))


def run_prepare(_args: argparse.Namespace) -> None:
    frame = prepare_dataset()
    print(f"Prepared {len(frame)} applicants written to {PROCESSED_PARQUET}")
    print(f"Provenance written to {REGISTRY_JSON}")


def run_train(args: argparse.Namespace) -> None:
    frame = load_training_data(args.data)
    model, metrics = train_model(frame, args.model_name)

    save_model(model, args.model_path)
    save_metrics(metrics, args.metrics_path)

    print(f"Trained {args.model_name} on {len(frame)} applicants")
    print(f"Model written to {args.model_path}")
    print(f"Metrics written to {args.metrics_path}")
    _report(metrics)


def run_evaluate(args: argparse.Namespace) -> None:
    frame = load_training_data(args.data)
    split = split_holdout(frame)
    model = load_model(args.model_path)

    metrics = score_model(model, split.holdout_features, split.holdout_target)

    print(f"Evaluated {args.model_path} on {len(split.holdout_target)} held-out applicants")
    _report(metrics)


def run_predict(args: argparse.Namespace) -> None:
    features = load_features_to_score(args.input_path)
    model = load_model(args.model_path)

    scored = predict_applicants(model, features)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_path, index=False)

    approved = int(scored["LoanApprovedPrediction"].sum())
    print(f"Scored {len(scored)} applicants, {approved} predicted approved")
    print(f"Predictions written to {args.output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit-risk",
        description="Train, evaluate, and apply the loan default risk model.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser(
        "prepare", help="Repair and convert the raw CSV to a typed parquet dataset"
    )
    prepare.set_defaults(handler=run_prepare)

    train = subcommands.add_parser("train", help="Train a model and write it to disk")
    train.add_argument("--data", type=Path, default=None, help="defaults to the prepared dataset")
    train.add_argument("--model-name", choices=sorted(MODEL_BUILDERS), default=DEFAULT_MODEL_NAME)
    train.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    train.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    train.set_defaults(handler=run_train)

    evaluate = subcommands.add_parser(
        "evaluate", help="Score a saved model on the held-out applicants"
    )
    evaluate.add_argument(
        "--data", type=Path, default=None, help="defaults to the prepared dataset"
    )
    evaluate.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    evaluate.set_defaults(handler=run_evaluate)

    predict = subcommands.add_parser("predict", help="Score new applicants from a CSV")
    predict.add_argument("--input-path", type=Path, required=True)
    predict.add_argument("--output-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    predict.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    predict.set_defaults(handler=run_predict)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
