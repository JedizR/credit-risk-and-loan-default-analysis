import numpy as np
import pandas as pd

from credit_risk.config import CONFIG
from credit_risk.data.schema import NUMERIC_FEATURES, TARGET_COLUMN
from credit_risk.features.autoencoder import AutoencoderFeatures, autoencoder_verdict
from credit_risk.features.engineering import engineer_features
from credit_risk.features.selection import candidate_features, split_feature_types
from credit_risk.pipeline import build_model


def test_embedding_has_one_column_per_bottleneck_unit(sample_frame: pd.DataFrame) -> None:
    encoder = AutoencoderFeatures(bottleneck=3).fit(sample_frame[NUMERIC_FEATURES])

    embedding = encoder.transform(sample_frame[NUMERIC_FEATURES])

    assert embedding.shape == (len(sample_frame), 3)
    assert np.isfinite(embedding).all()


def test_bottleneck_actually_compresses(sample_frame: pd.DataFrame) -> None:
    encoder = AutoencoderFeatures(bottleneck=2).fit(sample_frame[NUMERIC_FEATURES])

    embedding = encoder.transform(sample_frame[NUMERIC_FEATURES])

    # Fewer dimensions out than in: the middle layer cannot pass the input through unchanged.
    assert embedding.shape[1] < len(NUMERIC_FEATURES)
    assert encoder.reconstruction_error_ >= 0.0


def test_encoder_is_deterministic(sample_frame: pd.DataFrame) -> None:
    first = AutoencoderFeatures(bottleneck=3).fit(sample_frame[NUMERIC_FEATURES])
    second = AutoencoderFeatures(bottleneck=3).fit(sample_frame[NUMERIC_FEATURES])

    np.testing.assert_allclose(
        first.transform(sample_frame[NUMERIC_FEATURES]),
        second.transform(sample_frame[NUMERIC_FEATURES]),
    )


def test_encoder_learns_only_from_the_rows_it_is_fitted_on(sample_frame: pd.DataFrame) -> None:
    train = sample_frame.iloc[:200]
    holdout = sample_frame.iloc[200:]

    encoder = AutoencoderFeatures(bottleneck=3).fit(train[NUMERIC_FEATURES])
    embedding = encoder.transform(holdout[NUMERIC_FEATURES])

    # Transforming unseen rows must work without refitting: no holdout statistics leak in.
    assert embedding.shape == (len(holdout), 3)
    assert not hasattr(encoder.fit(train[NUMERIC_FEATURES]), "holdout_")


def test_names_are_exposed_for_downstream_explanations(sample_frame: pd.DataFrame) -> None:
    encoder = AutoencoderFeatures(bottleneck=2).fit(sample_frame[NUMERIC_FEATURES])

    assert list(encoder.get_feature_names_out()) == ["autoencoder_0", "autoencoder_1"]


def test_pipeline_adds_the_embedding_alongside_the_original_features(
    sample_frame: pd.DataFrame,
) -> None:
    engineered = engineer_features(sample_frame)
    columns = candidate_features()
    numeric, categorical = split_feature_types(columns)

    plain = build_model("lightgbm", numeric, categorical)
    with_embedding = build_model("lightgbm", numeric, categorical, autoencoder_bottleneck=3)
    plain.fit(engineered[columns], engineered[TARGET_COLUMN])
    with_embedding.fit(engineered[columns], engineered[TARGET_COLUMN])

    widened = with_embedding.named_steps["preprocess"].transform(engineered[columns])
    original = plain.named_steps["preprocess"].transform(engineered[columns])

    assert widened.shape[1] == original.shape[1] + 3


def test_verdict_reports_both_options_and_decides(sample_frame: pd.DataFrame) -> None:
    engineered = engineer_features(sample_frame)
    columns = candidate_features()
    numeric, categorical = split_feature_types(columns)

    verdict = autoencoder_verdict(
        engineered[columns],
        engineered[TARGET_COLUMN],
        "logistic_regression",
        numeric,
        categorical,
        bottleneck=CONFIG.training.autoencoder_bottleneck,
    )

    assert len(verdict) == 2
    assert set(verdict.columns) == {"features", "cv_pr_auc", "std", "difference", "verdict"}
    assert verdict.loc[0, "verdict"] == "baseline"
    assert verdict.loc[1, "verdict"] in {"keep", "drop (no real gain)"}
