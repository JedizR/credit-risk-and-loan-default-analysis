from collections.abc import Callable
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.config import CONFIG
from credit_risk.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MISSING_CATEGORY = "Missing"

MODEL_BUILDERS: dict[str, Callable[..., ClassifierMixin]] = {
    "logistic_regression": lambda **params: LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=CONFIG.seed,
        **params,
    ),
    "random_forest": lambda **params: RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=CONFIG.seed,
        **params,
    ),
    "gradient_boosting": lambda **params: HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=CONFIG.seed,
        **params,
    ),
    "lightgbm": lambda **params: LGBMClassifier(
        class_weight="balanced",
        random_state=CONFIG.seed,
        verbose=-1,
        **params,
    ),
}

DEFAULT_MODEL_NAME = CONFIG.training.model_name


class UnknownModelError(ValueError):
    def __init__(self, model_name: str) -> None:
        available = ", ".join(sorted(MODEL_BUILDERS))
        super().__init__(f"Unknown model '{model_name}'. Available models: {available}")


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = (
        CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    )

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

    # A selected subset may contain no columns of one kind; an empty branch would fail to fit.
    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_steps, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_steps, categorical_features))
    return ColumnTransformer(transformers)


def build_model(
    model_name: str = DEFAULT_MODEL_NAME,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    if model_name not in MODEL_BUILDERS:
        raise UnknownModelError(model_name)

    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("classifier", MODEL_BUILDERS[model_name](**(params or {}))),
        ]
    )
