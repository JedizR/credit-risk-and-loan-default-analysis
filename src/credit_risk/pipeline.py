from collections.abc import Callable

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_SEED = 42
MISSING_CATEGORY = "Missing"

MODEL_BUILDERS: dict[str, Callable[[], ClassifierMixin]] = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
}

DEFAULT_MODEL_NAME = "logistic_regression"


class UnknownModelError(ValueError):
    def __init__(self, model_name: str) -> None:
        available = ", ".join(sorted(MODEL_BUILDERS))
        super().__init__(f"Unknown model '{model_name}'. Available models: {available}")


def build_preprocessor() -> ColumnTransformer:
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_steps, NUMERIC_FEATURES),
            ("categorical", categorical_steps, CATEGORICAL_FEATURES),
        ]
    )


def build_model(model_name: str = DEFAULT_MODEL_NAME) -> Pipeline:
    if model_name not in MODEL_BUILDERS:
        raise UnknownModelError(model_name)
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("classifier", MODEL_BUILDERS[model_name]()),
        ]
    )
