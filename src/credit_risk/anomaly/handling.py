import pandas as pd

from credit_risk.anomaly.detectors import outlier_flags
from credit_risk.config import CONFIG
from credit_risk.data.quality import IMPOSSIBLE_NEGATIVE_FEATURES
from credit_risk.data.schema import NUMERIC_FEATURES, TARGET_COLUMN

IQR_FENCE = 1.5
Z_SCORE_LIMIT = 3.0
CONSENSUS_VOTES = 2


def impossible_value_flags(frame: pd.DataFrame) -> pd.Series:
    """Flag rows with a physically impossible negative value (domain rule)."""
    flagged = pd.Series(False, index=frame.index)
    for column in IMPOSSIBLE_NEGATIVE_FEATURES:
        if column in frame.columns:
            flagged |= frame[column] < 0
    return flagged.rename("impossible")


def iqr_flags(frame: pd.DataFrame, columns: list[str] = NUMERIC_FEATURES) -> pd.Series:
    """Flag rows outside the 1.5x IQR fence on any numeric column."""
    flagged = pd.Series(False, index=frame.index)
    for column in columns:
        values = frame[column]
        lower, upper = values.quantile(0.25), values.quantile(0.75)
        span = upper - lower
        flagged |= (values < lower - IQR_FENCE * span) | (values > upper + IQR_FENCE * span)
    return flagged.rename("iqr")


def z_score_flags(frame: pd.DataFrame, columns: list[str] = NUMERIC_FEATURES) -> pd.Series:
    """Flag rows more than 3 standard deviations from the mean on any numeric column."""
    flagged = pd.Series(False, index=frame.index)
    for column in columns:
        values = frame[column]
        deviations = ((values - values.mean()) / values.std()).abs()
        flagged |= deviations > Z_SCORE_LIMIT
    return flagged.rename("zscore")


def all_outlier_flags(frame: pd.DataFrame, contamination: float | None = None) -> pd.DataFrame:
    """Every lens on 'unusual': domain rule, univariate statistics, and model-based detectors."""
    rule_based = pd.concat(
        [impossible_value_flags(frame), iqr_flags(frame), z_score_flags(frame)], axis=1
    )
    return pd.concat([rule_based, outlier_flags(frame, contamination)], axis=1)


def summarize_outliers(flags: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """How many rows each method flags, and whether those rows differ in approval rate."""
    base_rate = frame[TARGET_COLUMN].mean()
    rows = {}
    for method in flags.columns:
        flagged = flags[method]
        rows[method] = {
            "flagged": int(flagged.sum()),
            "share_pct": round(flagged.mean() * 100, 2),
            "approval_rate": round(frame.loc[flagged, TARGET_COLUMN].mean(), 3)
            if flagged.any()
            else float("nan"),
            "approval_lift": round(frame.loc[flagged, TARGET_COLUMN].mean() - base_rate, 3)
            if flagged.any()
            else float("nan"),
        }
    return pd.DataFrame(rows).T


def detector_agreement(flags: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Jaccard overlap — low agreement means the detectors disagree on what is odd.

    Two methods that flag nothing are treated as agreeing completely (overlap 1.0): they agree that
    nothing is odd.
    """
    methods = list(flags.columns)
    overlap = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for left in methods:
        for right in methods:
            union = (flags[left] | flags[right]).sum()
            intersection = (flags[left] & flags[right]).sum()
            overlap.loc[left, right] = round(intersection / union, 3) if union else 1.0
    return overlap


def consensus_outliers(flags: pd.DataFrame, min_votes: int = CONSENSUS_VOTES) -> pd.Series:
    """Flag rows that at least ``min_votes`` methods agree are outliers."""
    return (flags.sum(axis=1) >= min_votes).rename("consensus")


def list_outliers(frame: pd.DataFrame, flags: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """The flagged rows themselves, annotated with which methods fired."""
    flagged_rows = frame[mask].copy()
    flagged_rows["flagged_by"] = flags[mask].apply(
        lambda row: ", ".join(flags.columns[row]), axis=1
    )
    return flagged_rows


def remove_outliers(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Return the frame without the masked rows."""
    return frame[~mask]


def clean_training_frame(
    frame: pd.DataFrame, contamination: float | None = None, min_votes: int = CONSENSUS_VOTES
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop consensus outliers from a training frame, returning the survivors and the mask.

    Only ever applied to training rows: removing holdout rows would flatter the evaluation.
    """
    contamination = contamination or CONFIG.training.outlier_contamination
    flags = all_outlier_flags(frame, contamination)
    mask = consensus_outliers(flags, min_votes)
    return remove_outliers(frame, mask), mask
