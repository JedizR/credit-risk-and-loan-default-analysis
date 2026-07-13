import numpy as np
import pandas as pd
import pytest

from credit_risk.data import FEATURE_COLUMNS
from credit_risk.pipeline import (
    MODEL_BUILDERS,
    UnknownModelError,
    build_model,
    build_preprocessor,
)


def test_preprocessor_leaves_no_missing_values(sample_frame: pd.DataFrame) -> None:
    features = sample_frame[FEATURE_COLUMNS]
    assert features.isna().to_numpy().any()

    encoded = build_preprocessor().fit_transform(features)

    assert not np.isnan(np.asarray(encoded, dtype=float)).any()


def test_preprocessor_encodes_missing_education_as_its_own_category(
    sample_frame: pd.DataFrame,
) -> None:
    preprocessor = build_preprocessor().fit(sample_frame[FEATURE_COLUMNS])

    encoded_names = preprocessor.get_feature_names_out()

    assert any("Education_Missing" in name for name in encoded_names)


@pytest.mark.parametrize("model_name", sorted(MODEL_BUILDERS))
def test_every_registered_model_fits_and_scores(
    model_name: str, sample_frame: pd.DataFrame
) -> None:
    features = sample_frame[FEATURE_COLUMNS]
    target = sample_frame["LoanApproved"]

    model = build_model(model_name).fit(features, target)
    probabilities = model.predict_proba(features)[:, 1]

    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0


def test_build_model_rejects_unknown_name() -> None:
    with pytest.raises(UnknownModelError, match="neural_net"):
        build_model("neural_net")
