from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.config import CONFIG
from credit_risk.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_risk.features.autoencoder import AutoencoderFeatures
from credit_risk.models import get_model

MISSING_CATEGORY = "Missing"
DEFAULT_MODEL_NAME = CONFIG.training.model_name


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    autoencoder_bottleneck: int | None = None,
) -> ColumnTransformer:
    """Build the learned preprocessing: impute+scale numerics, impute+one-hot categoricals.

    A selected subset may contain no columns of one kind, so an empty branch is skipped rather than
    added (it would otherwise fail to fit). When ``autoencoder_bottleneck`` is set, an autoencoder
    embedding of the numeric columns is added as an *extra* view alongside them, not a replacement.
    Everything here is learned, so it is refit inside every cross-validation fold.
    """
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

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_steps, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_steps, categorical_features))
    if autoencoder_bottleneck and numeric_features:
        transformers.append(
            ("autoencoder", AutoencoderFeatures(autoencoder_bottleneck), numeric_features)
        )
    return ColumnTransformer(transformers)


def build_model(
    model_name: str = DEFAULT_MODEL_NAME,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    params: dict[str, Any] | None = None,
    autoencoder_bottleneck: int | None = None,
) -> Pipeline:
    """Assemble the full model: the preprocessor followed by the chosen classifier.

    Args:
        model_name: A registered model name (see ``credit_risk.models``).
        numeric_features: Numeric columns for the preprocessor, or None for the schema default.
        categorical_features: Categorical columns, or None for the schema default.
        params: Classifier hyperparameters, e.g. an Optuna study's best params.
        autoencoder_bottleneck: If set, add an autoencoder embedding branch of that width.

    Returns:
        An unfitted scikit-learn ``Pipeline`` of preprocessing then classifier.

    Raises:
        UnknownModelError: If ``model_name`` is not registered.
    """
    classifier = get_model(model_name).build(**(params or {}))
    return Pipeline(
        [
            (
                "preprocess",
                build_preprocessor(numeric_features, categorical_features, autoencoder_bottleneck),
            ),
            ("classifier", classifier),
        ]
    )
