import warnings

__version__ = "0.1.0"


def silence_library_warnings() -> None:
    """Mute benign third-party warnings so a run's own output stays legible.

    SHAP colormap notices, scikit-learn's feature-name checks, and the autoencoder MLP's
    ``ConvergenceWarning`` are all ``UserWarning`` subclasses that carry no signal for this
    project. The CLI and the notebooks call this so neither buries its own results under them;
    the test suite deliberately does not, so genuine warnings still surface there.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
