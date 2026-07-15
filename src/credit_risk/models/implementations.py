from typing import TYPE_CHECKING, Any

from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from credit_risk.config import CONFIG
from credit_risk.models.base import CreditModel

if TYPE_CHECKING:
    from optuna.trial import Trial


class LogisticRegressionModel(CreditModel):
    """L2-regularised logistic regression — the interpretable baseline."""

    name = "logistic_regression"

    def build_estimator(self, **params: Any) -> ClassifierMixin:
        settings = {"max_iter": 1000, "class_weight": "balanced", "random_state": CONFIG.seed}
        return LogisticRegression(**{**settings, **params})

    def search_space(self, trial: "Trial") -> dict[str, Any]:
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}


class RandomForestModel(CreditModel):
    """Bagged decision trees over bootstrapped samples."""

    name = "random_forest"

    def build_estimator(self, **params: Any) -> ClassifierMixin:
        settings = {"n_estimators": 300, "class_weight": "balanced", "random_state": CONFIG.seed}
        return RandomForestClassifier(**{**settings, **params})

    def search_space(self, trial: "Trial") -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }


class GradientBoostingModel(CreditModel):
    """Histogram gradient boosting, with native missing-value handling."""

    name = "gradient_boosting"

    def build_estimator(self, **params: Any) -> ClassifierMixin:
        settings = {"class_weight": "balanced", "random_state": CONFIG.seed}
        return HistGradientBoostingClassifier(**{**settings, **params})

    def search_space(self, trial: "Trial") -> dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        }


class LightGBMModel(CreditModel):
    """Gradient-boosted trees — the project default, fastest at the best measured PR-AUC."""

    name = "lightgbm"

    def build_estimator(self, **params: Any) -> ClassifierMixin:
        settings = {"class_weight": "balanced", "random_state": CONFIG.seed, "verbose": -1}
        return LGBMClassifier(**{**settings, **params})

    def search_space(self, trial: "Trial") -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
