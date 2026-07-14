from matplotlib.figure import Figure


def assert_figure_is_drawn(figure: object) -> None:
    """A Figure object proves nothing — assert something was actually drawn on it.

    A blank figure once passed `isinstance(fig, Figure)` for weeks: the SHAP dependence plot
    builds its own figure, so the empty one we created was being returned instead.
    """
    assert isinstance(figure, Figure)
    assert figure.axes, "figure has no axes"
    assert any(
        axis.collections or axis.lines or axis.patches or axis.images or axis.texts
        for axis in figure.axes
    ), "figure has axes but nothing was drawn on them"
