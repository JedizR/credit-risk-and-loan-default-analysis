import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

CONFIG_FILE = Path("config.toml")


@dataclass(frozen=True)
class Paths:
    raw_csv: Path = Path("data/raw/loan_risk_prediction_dataset.csv")
    processed_parquet: Path = Path("data/processed/applicants.parquet")
    registry_json: Path = Path("data/registry.json")
    model: Path = Path("models/model.joblib")
    metrics: Path = Path("reports/metrics.json")
    predictions: Path = Path("out/predictions.csv")
    figures: Path = Path("reports/figures")


@dataclass(frozen=True)
class Thresholds:
    """Domain cut-offs measured in the exploratory analysis."""

    low_credit_score: float = 580.0
    low_income: float = 40_000.0
    informative_missing_delta: float = 0.05
    signal_mutual_info: float = 0.01


@dataclass(frozen=True)
class Training:
    model_name: str = "lightgbm"
    holdout_fraction: float = 0.2
    cross_validation_folds: int = 5
    decision_threshold: float = 0.5
    false_approval_cost: float = 5.0
    false_rejection_cost: float = 1.0
    optuna_trials: int = 30
    outlier_contamination: float = 0.01


@dataclass(frozen=True)
class Config:
    seed: int = 42
    sensitive_features: tuple[str, ...] = ("Gender", "City")
    paths: Paths = field(default_factory=Paths)
    thresholds: Thresholds = field(default_factory=Thresholds)
    training: Training = field(default_factory=Training)

    def with_training(self, **overrides: object) -> "Config":
        return replace(self, training=replace(self.training, **overrides))

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path = CONFIG_FILE) -> Config:
    if not Path(path).exists():
        return Config()

    overrides = tomllib.loads(Path(path).read_text())
    return Config(
        seed=overrides.get("seed", Config.seed),
        sensitive_features=tuple(overrides.get("sensitive_features", Config.sensitive_features)),
        paths=Paths(**{k: Path(v) for k, v in overrides.get("paths", {}).items()}),
        thresholds=Thresholds(**overrides.get("thresholds", {})),
        training=Training(**overrides.get("training", {})),
    )


CONFIG = load_config()
