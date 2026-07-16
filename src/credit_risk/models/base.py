from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from sklearn.base import ClassifierMixin

if TYPE_CHECKING:
    from optuna.trial import Trial

CLASSIFIER_METHODS = ("fit", "predict", "predict_proba", "get_params", "set_params")


@runtime_checkable
class ClassifierInterface(Protocol):
    """The scikit-learn classifier surface the pipeline, cross-validation and calibration rely on.

    Any model — including one written from scratch rather than wrapping a scikit-learn estimator —
    must expose these callables so it fits inside the ``Pipeline``, is clonable for
    cross-validation, and can be wrapped by ``CalibratedClassifierCV``.
    """

    def fit(self, features: Any, target: Any) -> Any: ...
    def predict(self, features: Any) -> Any: ...
    def predict_proba(self, features: Any) -> Any: ...
    def get_params(self, deep: bool = True) -> dict: ...
    def set_params(self, **params: Any) -> Any: ...


class CreditModel(ABC):
    """A named classifier the pipeline can build and Optuna can tune.

    A concrete model binds three things together in one place: a stable ``name`` used to select it
    from configuration, how to build its estimator (with the project's shared defaults — balanced
    class weights and the global seed), and the hyperparameter space to search. Adding a model is
    therefore adding one subclass.

    A genuinely new model — written from scratch rather than wrapping a scikit-learn estimator —
    implements :meth:`build_estimator` and :meth:`search_space`. The concrete :meth:`build` is the
    guardrail: it constructs the estimator and refuses anything that does not expose the classifier
    interface (:data:`CLASSIFIER_METHODS`) the rest of the project depends on.
    """

    name: str

    @abstractmethod
    def build_estimator(self, **params: Any) -> ClassifierMixin:
        """Build a fresh, unfitted estimator.

        Args:
            **params: Hyperparameters overriding or extending the model's defaults, typically the
                best parameters found by tuning.

        Returns:
            An unfitted classifier exposing the scikit-learn classifier interface
            (:class:`ClassifierInterface`).
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

    def build(self, **params: Any) -> ClassifierMixin:
        """Build the estimator and assert it satisfies the classifier interface the project needs.

        This is the guardrail for a model implemented from scratch: whatever
        :meth:`build_estimator` returns must ``fit``, ``predict``, ``predict_proba`` and expose
        ``get_params`` and ``set_params`` so it works inside the sklearn ``Pipeline``,
        cross-validation and calibration.

        Args:
            **params: Hyperparameters forwarded to :meth:`build_estimator`.

        Returns:
            The validated, unfitted estimator.

        Raises:
            TypeError: If the built estimator is missing any required classifier method.
        """
        estimator = self.build_estimator(**params)
        missing = [
            name for name in CLASSIFIER_METHODS if not callable(getattr(estimator, name, None))
        ]
        if missing:
            raise TypeError(
                f"{type(self).__name__}.build_estimator() returned an object missing required "
                f"classifier methods: {', '.join(missing)}. A CreditModel must build a "
                f"scikit-learn-compatible classifier (see ClassifierInterface)."
            )
        return estimator
