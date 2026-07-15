import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.config import CONFIG

RECONSTRUCTION_LAYERS = 2


class AutoencoderFeatures(BaseEstimator, TransformerMixin):
    """A bottleneck network trained to reconstruct its own input.

    scikit-learn ships no autoencoder, but an MLP fitted to predict X from X *is* one: the narrow
    middle layer cannot pass the input through unchanged, so it must compress it, and those
    activations are the embedding. Building it this way keeps `torch` — and roughly two gigabytes
    of image — out of the project for a technique that has not yet earned its place.

    Everything here is learned (the imputer's medians, the scaler's statistics, the network's
    weights), so it must live inside the model pipeline and be refit on each training fold.
    """

    def __init__(self, bottleneck: int = 3, hidden: int = 8, max_iter: int = 500) -> None:
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.max_iter = max_iter

    def fit(self, features: pd.DataFrame, target: pd.Series | None = None) -> "AutoencoderFeatures":  # noqa: ARG002
        self.preprocess_ = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        encoded = self.preprocess_.fit_transform(features)

        self.network_ = MLPRegressor(
            hidden_layer_sizes=(self.hidden, self.bottleneck, self.hidden),
            activation="relu",
            random_state=CONFIG.seed,
            max_iter=self.max_iter,
            early_stopping=False,
        )
        self.network_.fit(encoded, encoded)

        reconstructed = self.network_.predict(encoded)
        self.reconstruction_error_ = float(np.mean((encoded - reconstructed) ** 2))
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        activations = self.preprocess_.transform(features)
        weights = self.network_.coefs_[:RECONSTRUCTION_LAYERS]
        biases = self.network_.intercepts_[:RECONSTRUCTION_LAYERS]

        for weight, bias in zip(weights, biases, strict=True):
            activations = np.maximum(activations @ weight + bias, 0.0)
        return activations

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:  # noqa: ARG002
        return np.asarray([f"autoencoder_{index}" for index in range(self.bottleneck)])


def autoencoder_verdict(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    bottleneck: int | None = None,
) -> pd.DataFrame:
    """Cross-validated score with and without the embedding — the measurement that decides it.

    The embedding is only worth its complexity if it beats the engineered features it is added
    to. Anything inside one standard error is a tie, and a tie loses: the simpler model wins.

    Note:
        ``cross_validated_score`` and ``DEFAULT_MODEL_NAME`` are imported inside the function, not
        at module scope: ``pipeline`` imports this module for the optional embedding branch, so
        importing it back at the top would form a cycle.
    """
    from credit_risk.evaluation import cross_validated_score
    from credit_risk.pipeline import DEFAULT_MODEL_NAME

    model_name = model_name or DEFAULT_MODEL_NAME
    bottleneck = bottleneck or CONFIG.training.autoencoder_bottleneck

    without = cross_validated_score(
        features, target, model_name, numeric_features, categorical_features
    )
    with_embedding = cross_validated_score(
        features,
        target,
        model_name,
        numeric_features,
        categorical_features,
        autoencoder_bottleneck=bottleneck,
    )

    difference = with_embedding["mean"] - without["mean"]
    standard_error = without["std"] / np.sqrt(CONFIG.training.cross_validation_folds)

    return pd.DataFrame(
        [
            {
                "features": "engineered only",
                "cv_pr_auc": round(without["mean"], 4),
                "std": round(without["std"], 4),
            },
            {
                "features": f"engineered + autoencoder({bottleneck})",
                "cv_pr_auc": round(with_embedding["mean"], 4),
                "std": round(with_embedding["std"], 4),
            },
        ]
    ).assign(
        difference=[0.0, round(difference, 4)],
        verdict=["baseline", "keep" if difference > standard_error else "drop (no real gain)"],
    )
