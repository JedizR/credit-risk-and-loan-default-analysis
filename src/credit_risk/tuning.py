from collections.abc import Callable
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
from credit_risk.pipeline import DEFAULT_MODEL_NAME, MODEL_BUILDERS, UnknownModelError

SEARCH_SPACES: dict[str, Callable[[Trial], dict[str, Any]]] = {
    "logistic_regression": lambda trial: {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
    },
    "random_forest": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    },
    "gradient_boosting": lambda trial: {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
    },
    "lightgbm": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    },
}


def suggest_params(trial: Trial, model_name: str) -> dict[str, Any]:
    if model_name not in SEARCH_SPACES:
        raise UnknownModelError(model_name)
    return SEARCH_SPACES[model_name](trial)


def tune_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str = DEFAULT_MODEL_NAME,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    trials: int | None = None,
) -> optuna.Study:
    """Search hyperparameters against cross-validated PR-AUC.

    Every trial is scored by cross-validation *inside* the objective, so the preprocessing is
    refit on each fold and the holdout is never touched during the search.
    """
    trials = trials or CONFIG.training.optuna_trials
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: Trial) -> float:
        params = suggest_params(trial, model_name)
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
    return figure


def plot_param_importances(study: optuna.Study) -> Figure:
    """Which hyperparameters actually moved the score."""
    importances = optuna.importance.get_param_importances(study)
    ordered = pd.Series(importances).sort_values()

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.barh(ordered.index, ordered.to_numpy(), color="#4c72b0", alpha=0.85)
    axis.set(xlabel="importance", title="Hyperparameter importance")
    figure.tight_layout()
    return figure


def best_params(study: optuna.Study, model_name: str = DEFAULT_MODEL_NAME) -> dict[str, Any]:
    if model_name not in MODEL_BUILDERS:
        raise UnknownModelError(model_name)
    return dict(study.best_params)
