import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from credit_risk.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from credit_risk.eda.profile import RANDOM_STATE

APPROVED_PALETTE = {0: "#b0b0b0", 1: "#c44e52"}


def plot_numeric_distributions(
    frame: pd.DataFrame, columns: list[str] = NUMERIC_FEATURES
) -> Figure:
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.5))
    for column, axis in zip(columns, np.atleast_1d(axes), strict=True):
        sns.histplot(data=frame, x=column, kde=True, ax=axis)
        axis.set_title(column)
        axis.set_xlabel("")
    fig.tight_layout()
    return fig


def plot_categorical_counts(
    frame: pd.DataFrame, columns: list[str] = CATEGORICAL_FEATURES
) -> Figure:
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.5))
    for column, axis in zip(columns, np.atleast_1d(axes), strict=True):
        order = frame[column].value_counts().index
        sns.countplot(data=frame, y=column, order=order, ax=axis)
        axis.set_title(column)
        axis.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_missingness(frame: pd.DataFrame) -> Figure:
    missing_pct = frame.isna().mean().mul(100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    fig, axis = plt.subplots(figsize=(7, 0.5 * len(missing_pct) + 1))
    sns.barplot(x=missing_pct.to_numpy(), y=missing_pct.index, color="#c44e52", ax=axis)
    axis.bar_label(axis.containers[0], fmt="%.1f%%", padding=3)
    axis.set(xlabel="missing (%)", ylabel="", title="Missing values by column")
    fig.tight_layout()
    return fig


def plot_missingness_mechanism(frame: pd.DataFrame, target: str = TARGET_COLUMN) -> Figure:
    base_rate = frame[target].mean()
    columns = [c for c in frame.columns[frame.isna().any()] if c != target]
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.8))
    for column, axis in zip(columns, np.atleast_1d(axes), strict=True):
        missing = frame[column].isna()
        rates = [frame.loc[~missing, target].mean(), frame.loc[missing, target].mean()]
        axis.bar(["present", "missing"], rates, color=["#4c72b0", "#c44e52"])
        axis.axhline(base_rate, ls="--", c="k", lw=1)
        axis.set_title(f"{column} ({missing.mean() * 100:.1f}% missing)")
        axis.set_ylabel("approval rate")
    fig.suptitle("Approval rate: value present vs missing (dashed = base rate)")
    fig.tight_layout()
    return fig


def plot_boxplots_by_target(
    frame: pd.DataFrame, columns: list[str] = NUMERIC_FEATURES, target: str = TARGET_COLUMN
) -> Figure:
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4))
    for column, axis in zip(columns, np.atleast_1d(axes), strict=True):
        sns.boxplot(data=frame, x=target, y=column, hue=target, legend=False, ax=axis)
        axis.set_xlabel("")
    fig.tight_layout()
    return fig


def plot_correlation_heatmaps(
    frame: pd.DataFrame, columns: list[str] = NUMERIC_FEATURES, target: str = TARGET_COLUMN
) -> Figure:
    features = [*columns, target]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for method, axis in zip(["pearson", "spearman"], axes, strict=True):
        correlation = frame[features].corr(method=method)
        mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
        sns.heatmap(
            correlation,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            square=True,
            ax=axis,
        )
        axis.set_title(f"{method.title()} correlation")
    fig.tight_layout()
    return fig


def plot_approval_rate_by_category(
    frame: pd.DataFrame, columns: list[str] = CATEGORICAL_FEATURES, target: str = TARGET_COLUMN
) -> Figure:
    base_rate = frame[target].mean()
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4))
    for column, axis in zip(columns, np.atleast_1d(axes), strict=True):
        rate = frame.groupby(column, observed=True)[target].mean().sort_values(ascending=False)
        sns.barplot(x=rate.to_numpy(), y=rate.index, hue=rate.index, legend=False, ax=axis)
        axis.axvline(base_rate, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="approval rate", ylabel="", title=column)
    fig.tight_layout()
    return fig


def plot_tsne(
    frame: pd.DataFrame,
    target: str = TARGET_COLUMN,
    numeric: list[str] = NUMERIC_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
    perplexity: float = 30.0,
) -> Figure:
    numeric_block = frame[numeric].fillna(frame[numeric].median())
    encoded = pd.get_dummies(frame[categorical].astype("object").fillna("Missing"))
    standardized = StandardScaler().fit_transform(
        np.hstack([numeric_block.to_numpy(), encoded.to_numpy(dtype=float)])
    )
    embedding = TSNE(
        n_components=2, perplexity=perplexity, init="pca", random_state=RANDOM_STATE
    ).fit_transform(standardized)

    fig, axis = plt.subplots(figsize=(6.5, 6))
    for label, color in APPROVED_PALETTE.items():
        mask = (frame[target] == label).to_numpy()
        axis.scatter(
            embedding[mask, 0], embedding[mask, 1], s=6, c=color, label=str(label), alpha=0.6
        )
    axis.legend(title=target)
    axis.set_title("t-SNE of applicants (mixed features standardized)")
    fig.tight_layout()
    return fig


def plot_pca_scree(frame: pd.DataFrame, columns: list[str] = NUMERIC_FEATURES) -> Figure:
    standardized = StandardScaler().fit_transform(frame[columns].fillna(frame[columns].median()))
    variance_ratio = PCA(random_state=RANDOM_STATE).fit(standardized).explained_variance_ratio_

    fig, axis = plt.subplots(figsize=(6, 4))
    components = range(1, len(variance_ratio) + 1)
    axis.plot(components, variance_ratio.cumsum(), marker="o", label="cumulative")
    axis.bar(components, variance_ratio, alpha=0.4, label="per component")
    axis.set(xlabel="component", ylabel="explained variance", title="PCA scree (numeric features)")
    axis.legend()
    fig.tight_layout()
    return fig
