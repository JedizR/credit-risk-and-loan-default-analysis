import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from credit_risk.eda import plots  # noqa: E402
from tests.plot_assertions import assert_figure_is_drawn  # noqa: E402

PLOT_FUNCTIONS = [
    plots.plot_numeric_distributions,
    plots.plot_categorical_counts,
    plots.plot_missingness,
    plots.plot_missingness_mechanism,
    plots.plot_boxplots_by_target,
    plots.plot_correlation_heatmaps,
    plots.plot_approval_rate_by_category,
    plots.plot_pca_scree,
]


@pytest.mark.parametrize("plot_function", PLOT_FUNCTIONS)
def test_plot_returns_a_figure(plot_function, sample_frame: pd.DataFrame) -> None:
    figure = plot_function(sample_frame)

    assert_figure_is_drawn(figure)
    plt.close(figure)


def test_tsne_plot_embeds_every_applicant(sample_frame: pd.DataFrame) -> None:
    figure = plots.plot_tsne(sample_frame, perplexity=15.0)

    plotted_points = sum(len(collection.get_offsets()) for collection in figure.axes[0].collections)
    assert plotted_points == len(sample_frame)
    plt.close(figure)
