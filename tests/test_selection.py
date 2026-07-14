import pandas as pd
import pytest

from credit_risk.config import CONFIG
from credit_risk.data.schema import TARGET_COLUMN
from credit_risk.features.engineering import engineer_features
from credit_risk.features.selection import (
    FAMILIES,
    _source_feature,
    candidate_features,
    model_gain_ranking,
    mutual_information_ranking,
    permutation_ranking,
    rank_features,
    select_features,
    split_feature_types,
)


@pytest.fixture
def engineered(sample_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    frame = engineer_features(sample_frame)
    columns = candidate_features()
    return frame[columns], frame[TARGET_COLUMN]


def test_candidates_exclude_sensitive_features() -> None:
    for sensitive in CONFIG.sensitive_features:
        assert sensitive not in candidate_features()


def test_split_feature_types_partitions_every_column() -> None:
    columns = candidate_features()
    numeric, categorical = split_feature_types(columns)

    assert set(numeric) | set(categorical) == set(columns)
    assert not set(numeric) & set(categorical)


def test_one_hot_columns_map_back_to_their_source_feature() -> None:
    sources = ["EmploymentType", "credit_band", "Income", "income_missing"]

    assert _source_feature("categorical__EmploymentType_Unemployed", sources) == "EmploymentType"
    assert _source_feature("categorical__credit_band_prime", sources) == "credit_band"
    assert _source_feature("numeric__Income", sources) == "Income"
    assert _source_feature("numeric__income_missing", sources) == "income_missing"


def test_each_family_scores_every_candidate_feature(engineered) -> None:
    features, target = engineered

    filter_scores = mutual_information_ranking(features, target)
    embedded_scores = model_gain_ranking(features, target)
    wrapper_scores = permutation_ranking(features, target, "logistic_regression")

    for scores in (filter_scores, embedded_scores, wrapper_scores):
        assert set(scores.index) == set(features.columns)


def test_permutation_is_a_wrapper_measuring_held_out_loss(engineered) -> None:
    features, target = engineered

    scores = permutation_ranking(features, target, "logistic_regression")

    # A useful feature loses score when shuffled; a useless one hovers around zero.
    assert scores["CreditScore"] > 0
    assert scores.max() > abs(scores.min())


def test_ranking_has_one_column_per_family_plus_consensus(engineered) -> None:
    features, target = engineered

    ranking = rank_features(features, target, "logistic_regression")

    for family in FAMILIES:
        assert family in ranking.columns
        assert f"{family}_rank" in ranking.columns
    assert ranking["consensus"].is_monotonic_decreasing
    assert ranking["consensus"].between(0, 1).all()


def test_consensus_puts_credit_score_signal_on_top(engineered) -> None:
    features, target = engineered

    ranking = rank_features(features, target, "logistic_regression")

    assert "CreditScore" in ranking.index[:3]


def test_selection_returns_a_subset_and_a_cv_curve(engineered) -> None:
    features, target = engineered

    selected, curve = select_features(features, target, model_name="logistic_regression")

    assert 0 < len(selected) <= len(features.columns)
    assert set(selected).issubset(features.columns)
    assert len(curve) == len(features.columns)
    assert curve["mean"].between(0, 1).all()


def test_selection_prefers_the_parsimonious_set(engineered) -> None:
    features, target = engineered
    ranking = rank_features(features, target, "logistic_regression")

    selected, curve = select_features(features, target, ranking, "logistic_regression")

    # Never larger than the outright best-scoring set: the one-standard-error rule trims it.
    assert len(selected) <= int(curve["mean"].idxmax())
