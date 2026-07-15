import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from credit_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from credit_risk.models import UnknownModelError  # noqa: E402
from credit_risk.pipeline import build_model  # noqa: E402
from credit_risk.tuning import (  # noqa: E402
    best_params,
    plot_param_importances,
    plot_tuning_history,
    tune_model,
    tuning_history,
)
from tests.plot_assertions import assert_figure_is_drawn  # noqa: E402


def test_tuning_rejects_an_unknown_model() -> None:
    import optuna

    with pytest.raises(UnknownModelError, match="neural_net"):
        tune_model(
            pd.DataFrame({"a": [1, 2]}),
            pd.Series([0, 1]),
            "neural_net",
            trials=1,
        )
    assert optuna is not None


def test_tuning_improves_or_matches_the_default_params(sample_frame: pd.DataFrame) -> None:
    features, target = sample_frame[FEATURE_COLUMNS], sample_frame[TARGET_COLUMN]

    study = tune_model(features, target, "logistic_regression", trials=4)

    assert len(study.trials) == 4
    assert 0.0 <= study.best_value <= 1.0


def test_best_params_can_rebuild_the_model(sample_frame: pd.DataFrame) -> None:
    features, target = sample_frame[FEATURE_COLUMNS], sample_frame[TARGET_COLUMN]
    study = tune_model(features, target, "logistic_regression", trials=3)

    params = best_params(study, "logistic_regression")
    model = build_model("logistic_regression", params=params)

    assert params["C"] == model.named_steps["classifier"].C


def test_tuning_is_reproducible_under_the_same_seed(sample_frame: pd.DataFrame) -> None:
    features, target = sample_frame[FEATURE_COLUMNS], sample_frame[TARGET_COLUMN]

    first = tune_model(features, target, "logistic_regression", trials=3)
    second = tune_model(features, target, "logistic_regression", trials=3)

    assert first.best_params == second.best_params


def test_history_and_plots(sample_frame: pd.DataFrame) -> None:
    features, target = sample_frame[FEATURE_COLUMNS], sample_frame[TARGET_COLUMN]
    study = tune_model(features, target, "logistic_regression", trials=4)

    history = tuning_history(study)
    assert len(history) == 4

    for figure in (plot_tuning_history(study), plot_param_importances(study)):
        assert_figure_is_drawn(figure)
        plt.close(figure)
