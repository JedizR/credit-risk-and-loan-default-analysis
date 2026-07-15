from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline

from credit_risk.config import CONFIG

TOP_REASONS = 3
READABLE_NAMES = {
    "CreditScore": "credit score",
    "EmploymentType": "employment",
    "Income": "income",
    "LoanAmount": "loan amount",
    "YearsExperience": "years of experience",
    "Age": "age",
    "Education": "education",
    "credit_score_missing": "missing credit score",
    "income_missing": "missing income",
    "is_unemployed": "unemployed",
    "low_credit_score": "low credit score",
    "low_income": "low income",
    "credit_score_x_employed": "credit score while employed",
    "income_x_employed": "income while employed",
    "debt_to_income": "debt-to-income ratio",
    "experience_per_age": "experience relative to age",
    "credit_band": "credit band",
}


@dataclass(frozen=True)
class Explanation:
    """SHAP contributions for a set of applicants, aligned to readable feature names."""

    values: np.ndarray
    features: pd.DataFrame
    feature_names: list[str]
    base_value: float


def _classifier(model: Pipeline):
    return model.named_steps["classifier"]


def _transform(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return model[:-1].transform(features)


def _encoded_names(model: Pipeline) -> list[str]:
    raw_names = model[:-1].get_feature_names_out()
    return [name.split("__", 1)[-1] for name in raw_names]


def _display_values(model: Pipeline, transformed: np.ndarray) -> np.ndarray:
    """Undo the scaling for display only.

    SHAP values are computed on the scaled matrix, but a plot axis reading "CreditScore = -1.5"
    is unreadable. The numeric block is inverted back to real credit scores and incomes; the
    one-hot columns are already 0/1 and stay as they are.
    """
    preprocess = model.named_steps["preprocess"]
    encoded = preprocess.get_feature_names_out()
    numeric_columns = [i for i, name in enumerate(encoded) if name.startswith("numeric__")]
    if not numeric_columns:
        return transformed

    scaler = preprocess.named_transformers_["numeric"].named_steps["scale"]
    display = transformed.copy()
    display[:, numeric_columns] = scaler.inverse_transform(transformed[:, numeric_columns])
    return display


def explain_model(model: Pipeline, features: pd.DataFrame) -> Explanation:
    """SHAP values for a tree model, computed on the transformed feature space.

    SHAP returns one array per class for a binary classifier; only the approval class is kept, so
    the returned values explain the probability of approval.
    """
    transformed = _transform(model, features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    explainer = shap.TreeExplainer(_classifier(model))
    values = explainer.shap_values(transformed)
    expected = explainer.expected_value

    if isinstance(values, list):
        values = values[1]
        expected = expected[1]
    elif values.ndim == 3:
        values = values[:, :, 1]
        expected = expected[1] if np.ndim(expected) else expected

    names = _encoded_names(model)
    frame = pd.DataFrame(_display_values(model, transformed), columns=names, index=features.index)
    return Explanation(np.asarray(values), frame, names, float(np.ravel(expected)[0]))


def plot_beeswarm(explanation: Explanation, max_display: int = 15) -> Figure:
    """Global view: which features matter, in which direction, for whom."""
    figure = plt.figure()
    shap.summary_plot(
        explanation.values,
        explanation.features,
        feature_names=explanation.feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.close(figure)
    return figure


def plot_importance_bar(explanation: Explanation, max_display: int = 15) -> Figure:
    """Mean absolute SHAP per feature, as a ranked bar chart."""
    figure = plt.figure()
    shap.summary_plot(
        explanation.values,
        explanation.features,
        feature_names=explanation.feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.close(figure)
    return figure


def plot_dependence(
    explanation: Explanation, feature: str, interaction_feature: str | None = None
) -> Figure:
    """How a feature's effect changes with its value, coloured by an interacting feature.

    This is where the soft-AND shows up: a rising credit score only earns approval while the
    applicant is employed, so the two colours separate into different curves.

    Note:
        ``shap.dependence_plot`` builds its own figure instead of drawing on the current axes, so
        the figure it actually drew is reclaimed from pyplot with ``plt.gcf()``.
    """
    shap.dependence_plot(
        feature,
        explanation.values,
        explanation.features,
        feature_names=explanation.feature_names,
        interaction_index=interaction_feature,
        show=False,
    )
    figure = plt.gcf()
    figure.set_size_inches(7.5, 5)
    plt.tight_layout()
    plt.close(figure)
    return figure


def global_importance(explanation: Explanation) -> pd.Series:
    """Mean |SHAP| per feature, ranked descending — the global importance ordering."""
    mean_absolute = np.abs(explanation.values).mean(axis=0)
    return (
        pd.Series(mean_absolute, index=explanation.feature_names)
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
    )


def local_contributions(explanation: Explanation, position: int) -> pd.DataFrame:
    """The signed contribution of every feature for a single applicant."""
    contributions = pd.DataFrame(
        {
            "value": explanation.features.iloc[position],
            "shap": explanation.values[position],
        }
    )
    contributions["direction"] = np.where(contributions["shap"] > 0, "raises", "lowers")
    return contributions.reindex(contributions["shap"].abs().sort_values(ascending=False).index)


def _readable(encoded_name: str) -> str:
    if encoded_name in READABLE_NAMES:
        return READABLE_NAMES[encoded_name]

    for source, label in READABLE_NAMES.items():
        if encoded_name.startswith(f"{source}_"):
            return f"{label} is {encoded_name[len(source) + 1 :].replace('_', ' ')}"
    return encoded_name.replace("_", " ")


def decision_reasons(
    model: Pipeline,
    features: pd.DataFrame,
    threshold: float | None = None,
    top_reasons: int = TOP_REASONS,
) -> pd.DataFrame:
    """A decision, a probability, and the reasons a human can actually read."""
    threshold = CONFIG.training.decision_threshold if threshold is None else threshold
    explanation = explain_model(model, features)
    probabilities = model.predict_proba(features)[:, 1]

    rows = []
    for position in range(len(features)):
        contributions = explanation.values[position]
        strongest = np.argsort(np.abs(contributions))[::-1][:top_reasons]
        approved = probabilities[position] >= threshold

        reasons = [
            f"{_readable(explanation.feature_names[column])} "
            f"({'+' if contributions[column] > 0 else '-'})"
            for column in strongest
        ]
        rows.append(
            {
                "decision": "approved" if approved else "declined",
                "probability": round(float(probabilities[position]), 3),
                "reasons": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows, index=features.index)


def top_features(explanation: Explanation, count: int = 2) -> list[str]:
    """The most influential encoded features, so plots never name a column that was selected out."""
    return list(global_importance(explanation).index[:count])
