from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from sklearn.base import ClassifierMixin

if TYPE_CHECKING:
    from optuna.trial import Trial


class CreditModel(ABC):
    """A named classifier the pipeline can build and Optuna can tune.

    A concrete model binds three things together in one place: a stable ``name`` used to select it
    from configuration, how to build its estimator (with the project's shared defaults — balanced
    class weights and the global seed), and the hyperparameter space to search. Adding a model is
    therefore adding one subclass, and a genuinely new model — one that is not a thin wrapper around
    a scikit-learn estimator — has this contract to implement.
    """

    name: str

    @abstractmethod
    def build_estimator(self, **params: Any) -> ClassifierMixin:
        """Build a fresh, unfitted estimator.

        Args:
            **params: Hyperparameters overriding or extending the model's defaults, typically the
                best parameters found by tuning.

        Returns:
            An unfitted scikit-learn classifier ready to drop into the pipeline.
        """

    @abstractmethod
    def search_space(self, trial: "Trial") -> dict[str, Any]:
        """Suggest one trial's hyperparameters.

        Args:
            trial: The Optuna trial that draws each hyperparameter value.

        Returns:
            A mapping of hyperparameter name to the value suggested for this trial, ready to pass to
            :meth:`build_estimator`.
        """
