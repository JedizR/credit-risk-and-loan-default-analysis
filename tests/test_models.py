import optuna
import pytest
from sklearn.base import ClassifierMixin

from credit_risk.models import MODELS, CreditModel, UnknownModelError, get_model


def test_registry_holds_the_four_models() -> None:
    assert set(MODELS) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "lightgbm",
    }


@pytest.mark.parametrize("name", sorted(MODELS))
def test_each_model_builds_an_unfitted_estimator(name: str) -> None:
    model = get_model(name)
    assert isinstance(model, CreditModel)
    assert model.name == name
    assert isinstance(model.build_estimator(), ClassifierMixin)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_search_space_suggests_a_non_empty_mapping(name: str) -> None:
    trial = optuna.create_study().ask()
    space = get_model(name).search_space(trial)
    assert isinstance(space, dict)
    assert space


def test_build_estimator_applies_overrides() -> None:
    estimator = get_model("random_forest").build_estimator(n_estimators=17)
    assert estimator.get_params()["n_estimators"] == 17


def test_get_model_rejects_an_unknown_name() -> None:
    with pytest.raises(UnknownModelError, match="neural_net"):
        get_model("neural_net")


@pytest.mark.parametrize("name", sorted(MODELS))
def test_build_returns_an_estimator_with_the_classifier_interface(name: str) -> None:
    estimator = get_model(name).build()
    for method in ("fit", "predict", "predict_proba", "get_params", "set_params"):
        assert callable(getattr(estimator, method, None))


def test_build_rejects_a_from_scratch_model_missing_predict_proba() -> None:
    class BrokenModel(CreditModel):
        name = "broken"

        def build_estimator(self, **params):  # noqa: ARG002
            return object()

        def search_space(self, trial):  # noqa: ARG002
            return {}

    with pytest.raises(TypeError, match="predict_proba"):
        BrokenModel().build()
