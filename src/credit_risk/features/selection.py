import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from credit_risk.config import CONFIG
from credit_risk.evaluation import SCORING, cross_validated_score
from credit_risk.features.engineering import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES
from credit_risk.pipeline import DEFAULT_MODEL_NAME, build_model

EMBEDDED_MODEL = "lightgbm"
PERMUTATION_REPEATS = 5
FAMILIES = ["mutual_information", "model_gain", "permutation"]


def split_feature_types(columns: list[str]) -> tuple[list[str], list[str]]:
    numeric = [column for column in columns if column in MODEL_NUMERIC_FEATURES]
    categorical = [column for column in columns if column in MODEL_CATEGORICAL_FEATURES]
    return numeric, categorical


def candidate_features() -> list[str]:
    """Every original and engineered feature except the ones excluded on fair-lending grounds."""
    everything = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
    return [column for column in everything if column not in CONFIG.sensitive_features]


def mutual_information_ranking(features: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Filter method: dependence between each feature and the target, model-free."""
    numeric, categorical = split_feature_types(list(features.columns))
    encoded = pd.DataFrame(index=features.index)
    for column in numeric:
        encoded[column] = features[column].fillna(features[column].median())
    for column in categorical:
        encoded[column] = features[column].astype("category").cat.codes

    ordered = numeric + categorical
    scores = mutual_info_classif(
        encoded[ordered],
        target,
        discrete_features=[False] * len(numeric) + [True] * len(categorical),
        random_state=CONFIG.seed,
    )
    return pd.Series(scores, index=ordered, name="mutual_information")


def model_gain_ranking(features: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Embedded method: total split gain a gradient-boosted tree earns from each feature.

    Gains are reported per encoded column, so one-hot columns are summed back onto the
    source feature they came from.
    """
    numeric, categorical = split_feature_types(list(features.columns))
    model = build_model(EMBEDDED_MODEL, numeric, categorical).fit(features, target)

    encoded_names = model.named_steps["preprocess"].get_feature_names_out()
    gains = model.named_steps["classifier"].booster_.feature_importance(importance_type="gain")

    sources = [_source_feature(name, numeric + categorical) for name in encoded_names]
    return pd.Series(gains, index=sources).groupby(level=0).sum().rename("model_gain")


def _source_feature(encoded_name: str, source_columns: list[str]) -> str:
    without_prefix = encoded_name.split("__", 1)[-1]
    if without_prefix in source_columns:
        return without_prefix

    candidates = [column for column in source_columns if without_prefix.startswith(f"{column}_")]
    return max(candidates, key=len) if candidates else without_prefix


def permutation_ranking(
    features: pd.DataFrame, target: pd.Series, model_name: str = DEFAULT_MODEL_NAME
) -> pd.Series:
    """Wrapper method: the PR-AUC a fitted model *loses* when a feature is shuffled.

    The shuffling is measured on a held-out fold the model never saw. Permuting the rows the
    model was fitted on would measure memorisation, not usefulness.
    """
    numeric, categorical = split_feature_types(list(features.columns))
    fit_features, validation_features, fit_target, validation_target = train_test_split(
        features,
        target,
        test_size=CONFIG.training.holdout_fraction,
        random_state=CONFIG.seed,
        stratify=target,
    )
    model = build_model(model_name, numeric, categorical).fit(fit_features, fit_target)

    importance = permutation_importance(
        model,
        validation_features,
        validation_target,
        scoring=SCORING,
        n_repeats=PERMUTATION_REPEATS,
        random_state=CONFIG.seed,
        n_jobs=-1,
    )
    return pd.Series(importance.importances_mean, index=features.columns, name="permutation")


def _normalised_rank(scores: pd.Series) -> pd.Series:
    """Map raw scores onto a common 0-1 scale so families with different units can be averaged."""
    if len(scores) < 2:
        return pd.Series(1.0, index=scores.index)
    ranks = scores.rank(ascending=False, method="average")
    return (len(scores) - ranks) / (len(scores) - 1)


def rank_features(
    features: pd.DataFrame, target: pd.Series, model_name: str = DEFAULT_MODEL_NAME
) -> pd.DataFrame:
    """Combine one filter, one embedded and one wrapper method into a consensus ranking.

    The three families disagree by construction — mutual information cannot see interactions,
    gain rewards features a tree happens to split on, and permutation measures real predictive
    loss. Averaging their normalised ranks keeps a feature that only one family likes from
    dominating, and a feature all three like rises to the top.
    """
    scores = pd.DataFrame(
        {
            "mutual_information": mutual_information_ranking(features, target),
            "model_gain": model_gain_ranking(features, target),
            "permutation": permutation_ranking(features, target, model_name),
        }
    ).reindex(features.columns)

    for family in FAMILIES:
        scores[f"{family}_rank"] = _normalised_rank(scores[family])
    scores["consensus"] = scores[[f"{family}_rank" for family in FAMILIES]].mean(axis=1)

    return scores.sort_values("consensus", ascending=False)


def selection_curve(
    features: pd.DataFrame,
    target: pd.Series,
    ranking: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
) -> pd.DataFrame:
    """Cross-validated score of the top-k features, for every k."""
    ordered = list(ranking.index)
    rows = []
    for size in range(1, len(ordered) + 1):
        subset = ordered[:size]
        numeric, categorical = split_feature_types(subset)
        scores = cross_validated_score(features[subset], target, model_name, numeric, categorical)
        rows.append(
            {
                "k": size,
                "added": subset[-1],
                "mean": round(scores["mean"], 4),
                "std": round(scores["std"], 4),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def select_features(
    features: pd.DataFrame,
    target: pd.Series,
    ranking: pd.DataFrame | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[list[str], pd.DataFrame]:
    """Keep the smallest feature set that is statistically as good as the best one.

    Picking the outright maximum overfits the selection itself, so the one-standard-error
    rule is applied: take the fewest features whose cross-validated score is still within one
    standard error of the best score.
    """
    ranking = rank_features(features, target, model_name) if ranking is None else ranking
    curve = selection_curve(features, target, ranking, model_name)

    best = curve["mean"].idxmax()
    standard_error = curve.loc[best, "std"] / np.sqrt(CONFIG.training.cross_validation_folds)
    within_one_error = curve[curve["mean"] >= curve.loc[best, "mean"] - standard_error]
    chosen = int(within_one_error.index.min())

    return list(ranking.index[:chosen]), curve


def plot_selection_curve(curve: pd.DataFrame, selected: int | None = None) -> Figure:
    """Cross-validated score as features are added, with the chosen set marked."""
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.errorbar(curve.index, curve["mean"], yerr=curve["std"], marker="o", capsize=3)

    best = int(curve["mean"].idxmax())
    axis.axvline(best, ls=":", c="grey", label=f"best score (k={best})")
    if selected:
        axis.axvline(selected, ls="--", c="#c44e52", label=f"selected (k={selected})")

    axis.set(
        xlabel="number of features (ranked by consensus)",
        ylabel="cross-validated PR-AUC",
        title="Feature selection curve",
    )
    axis.legend()
    figure.tight_layout()
    return figure
