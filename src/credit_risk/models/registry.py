from credit_risk.models.base import CreditModel
from credit_risk.models.implementations import (
    GradientBoostingModel,
    LightGBMModel,
    LogisticRegressionModel,
    RandomForestModel,
)

MODELS: dict[str, CreditModel] = {
    model.name: model
    for model in (
        LogisticRegressionModel(),
        RandomForestModel(),
        GradientBoostingModel(),
        LightGBMModel(),
    )
}


class UnknownModelError(ValueError):
    """Raised when a model name is not in the registry."""

    def __init__(self, model_name: str) -> None:
        available = ", ".join(sorted(MODELS))
        super().__init__(f"Unknown model '{model_name}'. Available models: {available}")


def get_model(name: str) -> CreditModel:
    """Look up a registered model by name.

    Args:
        name: A registered model name, e.g. ``"lightgbm"``.

    Returns:
        The :class:`CreditModel` registered under ``name``.

    Raises:
        UnknownModelError: If no model is registered under ``name``.
    """
    try:
        return MODELS[name]
    except KeyError:
        raise UnknownModelError(name) from None
