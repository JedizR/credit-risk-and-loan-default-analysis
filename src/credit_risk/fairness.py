import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from credit_risk.config import CONFIG

FOUR_FIFTHS = 0.8


def fairness_audit(classified: pd.DataFrame, protected: list[str] | None = None) -> pd.DataFrame:
    """How the model treats each protected group.

    Excluding a protected attribute from the features does not make a model fair: another column
    can act as a proxy for it. The only way to know is to hold the outcomes up against the groups
    and look.
    """
    protected = protected or list(CONFIG.sensitive_features)

    rows = []
    for attribute in protected:
        for group, subset in classified.groupby(attribute, observed=True):
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "applicants": len(subset),
                    "selection_rate": round((subset["predicted"] == 1).mean(), 3),
                    "error_rate": round(subset["is_error"].mean(), 3),
                    "false_positive_rate": round((subset["outcome"] == "false_positive").mean(), 3),
                    "false_negative_rate": round((subset["outcome"] == "false_negative").mean(), 3),
                }
            )
    return pd.DataFrame(rows).set_index(["attribute", "group"])


def disparate_impact(audit: pd.DataFrame) -> pd.DataFrame:
    """The four-fifths rule: the least-selected group must reach 80% of the most-selected one.

    This is the standard US fair-lending screen. A ratio below 0.8 is the point at which a
    regulator would ask the lender to justify itself.
    """
    rows = []
    for attribute, block in audit.groupby(level="attribute"):
        rates = block["selection_rate"]
        ratio = float(rates.min() / rates.max()) if rates.max() else 0.0
        gap = float(block["false_negative_rate"].max() - block["false_negative_rate"].min())
        rows.append(
            {
                "attribute": attribute,
                "lowest_selection_rate": rates.min(),
                "highest_selection_rate": rates.max(),
                "impact_ratio": round(ratio, 3),
                "passes_four_fifths": bool(ratio >= FOUR_FIFTHS),
                "false_negative_gap": round(gap, 3),
            }
        )
    return pd.DataFrame(rows).set_index("attribute")


def plot_fairness(audit: pd.DataFrame) -> Figure:
    """Selection rate and error rate side by side, per group, against the population average."""
    attributes = audit.index.get_level_values("attribute").unique()
    figure, axes = plt.subplots(len(attributes), 2, figsize=(12, 3.4 * len(attributes)))
    axes = np.atleast_2d(axes)

    for row, attribute in enumerate(attributes):
        block = audit.xs(attribute, level="attribute")
        for column, (metric, title) in enumerate(
            [("selection_rate", "approval rate"), ("error_rate", "error rate")]
        ):
            axis = axes[row, column]
            values = block[metric]
            sns.barplot(
                x=values.to_numpy(),
                y=values.index.astype(str),
                hue=values.index.astype(str),
                legend=False,
                ax=axis,
            )
            axis.axvline(values.mean(), ls="--", c="black", lw=1)
            axis.set(xlabel=title, ylabel="", title=f"{attribute} — {title}")

    figure.tight_layout()
    plt.close(figure)
    return figure


def impact_ratio_confidence(
    classified: pd.DataFrame, attribute: str, resamples: int = 500
) -> dict[str, float]:
    """Bootstrap interval for the impact ratio.

    With only a few hundred applicants per group, a ratio of 0.77 and a ratio of 0.85 can be the
    same underlying model seen through different samples. Reporting the point estimate alone
    would invite a conclusion the data cannot support — in either direction.
    """
    generator = np.random.default_rng(CONFIG.seed)
    ratios = []

    for _ in range(resamples):
        sample = classified.sample(
            len(classified), replace=True, random_state=generator.integers(1e9)
        )
        rates = sample.groupby(attribute, observed=True)["predicted"].mean()
        if len(rates) > 1 and rates.max() > 0:
            ratios.append(rates.min() / rates.max())

    observed = classified.groupby(attribute, observed=True)["predicted"].mean()
    return {
        "impact_ratio": round(float(observed.min() / observed.max()), 3),
        "lower_95": round(float(np.percentile(ratios, 2.5)), 3),
        "upper_95": round(float(np.percentile(ratios, 97.5)), 3),
        "passes_four_fifths": bool(observed.min() / observed.max() >= FOUR_FIFTHS),
        "conclusive": bool(np.percentile(ratios, 97.5) < FOUR_FIFTHS),
    }
