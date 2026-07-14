import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from credit_risk.data.quality import IMPOSSIBLE_NEGATIVE_FEATURES
from credit_risk.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN

RANDOM_STATE = 42
INFORMATIVE_MISSING_DELTA = 0.05
SIGNAL_MUTUAL_INFO = 0.01
SENSITIVE_FEATURES = ["Gender", "City"]


def overview(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "rows": len(frame),
            "columns": frame.shape[1],
            "duplicate_rows": int(frame.duplicated().sum()),
            "memory_kb": round(frame.memory_usage(deep=True).sum() / 1024, 1),
        }
    )


def numeric_summary(frame: pd.DataFrame, columns: list[str] = NUMERIC_FEATURES) -> pd.DataFrame:
    rows = {}
    for column in columns:
        series = frame[column]
        rows[column] = {
            "missing_pct": round(series.isna().mean() * 100, 2),
            "min": series.min(),
            "median": series.median(),
            "max": series.max(),
            "mean": round(series.mean(), 1),
            "std": round(series.std(), 1),
            "skew": round(series.skew(), 2),
            "kurtosis": round(series.kurt(), 2),
        }
    return pd.DataFrame(rows).T


def categorical_summary(
    frame: pd.DataFrame, columns: list[str] = CATEGORICAL_FEATURES
) -> pd.DataFrame:
    rows = {}
    for column in columns:
        series = frame[column]
        counts = series.value_counts()
        rows[column] = {
            "cardinality": series.nunique(),
            "missing_pct": round(series.isna().mean() * 100, 2),
            "mode": counts.index[0],
            "mode_pct": round(counts.iloc[0] / len(series) * 100, 1),
        }
    return pd.DataFrame(rows).T


def missingness(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.isna().sum()
    summary = pd.DataFrame(
        {"missing_count": counts, "missing_pct": (counts / len(frame) * 100).round(2)}
    )
    return summary[summary["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def missingness_mechanism(frame: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.DataFrame:
    base_rate = frame[target].mean()
    rows = {}
    for column in frame.columns[frame.isna().any()]:
        missing = frame[column].isna()
        rate_missing = frame.loc[missing, target].mean()
        rate_present = frame.loc[~missing, target].mean()
        delta = rate_missing - rate_present
        rows[column] = {
            "missing_pct": round(missing.mean() * 100, 2),
            "rate_present": round(rate_present, 3),
            "rate_missing": round(rate_missing, 3),
            "delta": round(delta, 3),
            "mechanism": "informative" if abs(delta) >= INFORMATIVE_MISSING_DELTA else "at random",
        }
    summary = pd.DataFrame(rows).T
    summary.attrs["base_rate"] = round(base_rate, 3)
    return summary


def outlier_summary(frame: pd.DataFrame, columns: list[str] = NUMERIC_FEATURES) -> pd.DataFrame:
    rows = {}
    for column in columns:
        series = frame[column].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        z_scores = (series - series.mean()) / series.std()
        rows[column] = {
            "iqr_low": int((series < q1 - 1.5 * iqr).sum()),
            "iqr_high": int((series > q3 + 1.5 * iqr).sum()),
            "z_over_3": int((z_scores.abs() > 3).sum()),
            "impossible_negative": int((frame[column] < 0).sum())
            if column in IMPOSSIBLE_NEGATIVE_FEATURES
            else 0,
        }
    return pd.DataFrame(rows).T


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x, y)
    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(confusion, correction=False)[0]
    n = confusion.to_numpy().sum()
    smallest_dimension = min(confusion.shape) - 1
    return float(np.sqrt((chi2 / n) / smallest_dimension))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    valid = values.notna()
    categories, values = categories[valid], values[valid]
    grand_mean = values.mean()
    between = sum(
        len(group) * (group.mean() - grand_mean) ** 2 for _, group in values.groupby(categories)
    )
    total = ((values - grand_mean) ** 2).sum()
    return float(np.sqrt(between / total)) if total else 0.0


def target_association(
    frame: pd.DataFrame,
    target: str = TARGET_COLUMN,
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
) -> pd.DataFrame:
    target_values = frame[target]
    encoded = pd.DataFrame(index=frame.index)
    for column in numeric:
        encoded[column] = frame[column].fillna(frame[column].median())
    for column in categorical:
        encoded[column] = frame[column].astype("category").cat.codes
    discrete = [False] * len(numeric) + [True] * len(categorical)
    scores = mutual_info_classif(
        encoded[numeric + categorical],
        target_values,
        discrete_features=discrete,
        random_state=RANDOM_STATE,
    )

    rows = []
    for feature, mutual_info in zip(numeric + categorical, scores, strict=True):
        if feature in numeric:
            present = frame[feature].notna()
            secondary = roc_auc_score(target_values[present], frame[feature][present])
            feature_type = "numeric"
        else:
            secondary = cramers_v(frame[feature], target_values)
            feature_type = "categorical"
        rows.append(
            {
                "feature": feature,
                "type": feature_type,
                "mutual_info": round(float(mutual_info), 4),
                "auc_or_cramers_v": round(float(secondary), 3),
                "signal": "signal" if mutual_info >= SIGNAL_MUTUAL_INFO else "noise",
            }
        )
    return pd.DataFrame(rows).set_index("feature").sort_values("mutual_info", ascending=False)


def approval_rate_by_band(
    frame: pd.DataFrame, feature: str, target: str = TARGET_COLUMN, bins: int = 5
) -> pd.DataFrame:
    bands = pd.qcut(frame[feature], bins, duplicates="drop")
    grouped = frame.groupby(bands, observed=True)[target].agg(["mean", "count"])
    return grouped.rename(columns={"mean": "approval_rate"}).round(3)


def feature_decision_table(frame: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.DataFrame:
    association = target_association(frame, target)
    mechanism = missingness_mechanism(frame, target)
    outliers = outlier_summary(frame)

    rows = {}
    for feature in association.index:
        is_sensitive = feature in SENSITIVE_FEATURES
        is_signal = association.loc[feature, "signal"] == "signal"
        missing_pct = (
            float(mechanism.loc[feature, "missing_pct"]) if feature in mechanism.index else 0.0
        )
        is_informative_missing = (
            feature in mechanism.index and mechanism.loc[feature, "mechanism"] == "informative"
        )
        impossible = (
            int(outliers.loc[feature, "impossible_negative"]) if feature in outliers.index else 0
        )
        rows[feature] = {
            "mutual_info": association.loc[feature, "mutual_info"],
            "missing_pct": missing_pct,
            "impossible_negatives": impossible,
            "recommendation": _recommend(
                is_sensitive, is_signal, is_informative_missing, impossible
            ),
        }
    return pd.DataFrame(rows).T


def _recommend(sensitive: bool, signal: bool, informative_missing: bool, impossible: int) -> str:
    if sensitive:
        return "exclude (sensitive attribute); audit for bias"
    if signal and informative_missing:
        return "keep (primary); missingness is itself informative"
    if signal:
        return "keep (predictive)"
    if impossible:
        return "repair impossible values; weak signal"
    return "drop candidate (noise)"
