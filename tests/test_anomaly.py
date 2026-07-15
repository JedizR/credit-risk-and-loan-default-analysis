import pandas as pd
import pytest

from credit_risk.anomaly.base import AnomalyDetector
from credit_risk.anomaly.detectors import (
    DETECTORS,
    UnknownDetectorError,
    build_detector,
    detect_outliers,
    get_detector,
)
from credit_risk.anomaly.handling import (
    all_outlier_flags,
    clean_training_frame,
    consensus_outliers,
    detector_agreement,
    impossible_value_flags,
    iqr_flags,
    list_outliers,
    remove_outliers,
    summarize_outliers,
    z_score_flags,
)


def test_build_detector_rejects_unknown_name() -> None:
    with pytest.raises(UnknownDetectorError, match="autoencoder"):
        build_detector("autoencoder")


def test_registry_holds_anomaly_detectors() -> None:
    assert set(DETECTORS) == {"isolation_forest", "one_class_svm"}
    for name, detector in DETECTORS.items():
        assert isinstance(detector, AnomalyDetector)
        assert get_detector(name) is detector


@pytest.mark.parametrize("detector_name", sorted(DETECTORS))
def test_every_detector_returns_an_index_aligned_mask(
    detector_name: str, sample_frame: pd.DataFrame
) -> None:
    flags = detect_outliers(sample_frame, detector_name, contamination=0.05)

    assert flags.index.equals(sample_frame.index)
    assert flags.dtype == bool
    assert 0 < flags.sum() < len(sample_frame)


def test_detectors_do_not_look_at_the_target(sample_frame: pd.DataFrame) -> None:
    shuffled = sample_frame.copy()
    shuffled["LoanApproved"] = 1 - shuffled["LoanApproved"]

    original = detect_outliers(sample_frame, "isolation_forest", contamination=0.05)
    flipped = detect_outliers(shuffled, "isolation_forest", contamination=0.05)

    assert original.equals(flipped)


def test_impossible_flags_catch_only_negatives(sample_frame: pd.DataFrame) -> None:
    frame = sample_frame.copy()
    frame.loc[frame.index[0], "Income"] = -1.0

    flags = impossible_value_flags(frame)

    assert flags.sum() == 1
    assert flags.iloc[0]


def test_iqr_and_zscore_flags_are_boolean_and_aligned(sample_frame: pd.DataFrame) -> None:
    for flags in (iqr_flags(sample_frame), z_score_flags(sample_frame)):
        assert flags.dtype == bool
        assert flags.index.equals(sample_frame.index)


def test_all_outlier_flags_has_one_column_per_method(sample_frame: pd.DataFrame) -> None:
    flags = all_outlier_flags(sample_frame, contamination=0.05)

    assert set(flags.columns) == {"impossible", "iqr", "zscore", *DETECTORS}


def test_summary_reports_counts_and_approval_lift(sample_frame: pd.DataFrame) -> None:
    flags = all_outlier_flags(sample_frame, contamination=0.05)

    summary = summarize_outliers(flags, sample_frame)

    assert set(summary.columns) == {"flagged", "share_pct", "approval_rate", "approval_lift"}
    assert summary.loc["isolation_forest", "flagged"] == flags["isolation_forest"].sum()


def test_agreement_matrix_is_symmetric_with_unit_diagonal(sample_frame: pd.DataFrame) -> None:
    flags = all_outlier_flags(sample_frame, contamination=0.05)

    overlap = detector_agreement(flags)

    assert (overlap.to_numpy().diagonal() == 1.0).all()
    assert overlap.equals(overlap.T)


def test_consensus_requires_more_than_one_vote(sample_frame: pd.DataFrame) -> None:
    flags = all_outlier_flags(sample_frame, contamination=0.05)

    consensus = consensus_outliers(flags, min_votes=2)

    assert (consensus <= flags.any(axis=1)).all()
    assert consensus.sum() <= flags.any(axis=1).sum()


def test_list_outliers_annotates_which_methods_fired(sample_frame: pd.DataFrame) -> None:
    flags = all_outlier_flags(sample_frame, contamination=0.05)
    mask = flags.any(axis=1)

    listed = list_outliers(sample_frame, flags, mask)

    assert len(listed) == mask.sum()
    assert listed["flagged_by"].str.len().gt(0).all()


def test_remove_outliers_drops_exactly_the_flagged_rows(sample_frame: pd.DataFrame) -> None:
    flags = all_outlier_flags(sample_frame, contamination=0.05)
    mask = flags.any(axis=1)

    survivors = remove_outliers(sample_frame, mask)

    assert len(survivors) == len(sample_frame) - mask.sum()
    assert not survivors.index.intersection(sample_frame[mask].index).size


def test_clean_training_frame_returns_survivors_and_mask(sample_frame: pd.DataFrame) -> None:
    cleaned, mask = clean_training_frame(sample_frame, contamination=0.05)

    assert len(cleaned) == len(sample_frame) - mask.sum()
    assert "LoanApproved" in cleaned.columns
