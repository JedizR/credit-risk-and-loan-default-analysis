from typing import Any

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from matplotlib.figure import Figure
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial

from credit_risk.config import CONFIG
from credit_risk.evaluation import cross_validated_score
from credit_risk.models import get_model
from credit_risk.pipeline import DEFAULT_MODEL_NAME


def tune_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str = DEFAULT_MODEL_NAME,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    trials: int | None = None,
) -> optuna.Study:
    """Search hyperparameters against cross-validated PR-AUC.

    Every trial is scored by cross-validation *inside* the objective, so the preprocessing is refit
    on each fold and the holdout is never touched during the search.

    Args:
        features: The training feature frame.
        target: The training labels aligned to ``features``.
        model_name: The registered model to tune.
        numeric_features: Numeric columns for the preprocessor, or None for the schema default.
        categorical_features: Categorical columns for the preprocessor, or None for the default.
        trials: Number of Optuna trials, or None for ``CONFIG.training.optuna_trials``.

    Returns:
        The completed Optuna study, whose ``best_params`` rebuild the tuned model.

    Raises:
        UnknownModelError: If ``model_name`` is not registered.
    """
    model = get_model(model_name)
    trials = trials or CONFIG.training.optuna_trials
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Trial) -> float:
        params = model.search_space(trial)
        scores = cross_validated_score(
            features, target, model_name, numeric_features, categorical_features, params
        )
        return scores["mean"]

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=CONFIG.seed),
        pruner=MedianPruner(),
        study_name=f"{model_name}-pr-auc",
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study


def tuning_history(study: optuna.Study) -> pd.DataFrame:
    return study.trials_dataframe(attrs=("number", "value", "params", "state"))


def plot_tuning_history(study: optuna.Study) -> Figure:
    """Did the search converge, and how much did it actually buy?"""
    values = [trial.value for trial in study.trials if trial.value is not None]
    running_best = pd.Series(values).cummax()

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.scatter(range(len(values)), values, s=18, alpha=0.5, label="trial")
    axis.plot(running_best, color="#c44e52", label="best so far")
    axis.set(
        xlabel="trial",
        ylabel="cross-validated PR-AUC",
        title=f"Optuna search ({study.study_name})",
    )
    axis.legend()
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_param_importances(study: optuna.Study) -> Figure:
    """Which hyperparameters actually moved the score."""
    importances = optuna.importance.get_param_importances(study)
    ordered = pd.Series(importances).sort_values()

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.barh(ordered.index, ordered.to_numpy(), color="#4c72b0", alpha=0.85)
    axis.set(xlabel="importance", title="Hyperparameter importance")
    figure.tight_layout()
    plt.close(figure)
    return figure


def best_params(study: optuna.Study, model_name: str = DEFAULT_MODEL_NAME) -> dict[str, Any]:
    get_model(model_name)
    return dict(study.best_params)
