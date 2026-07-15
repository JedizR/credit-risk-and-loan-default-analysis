from abc import ABC, abstractmethod

from sklearn.base import OutlierMixin


class AnomalyDetector(ABC):
    """A named unsupervised outlier detector the pipeline can build at a target contamination.

    A concrete detector binds a stable ``name`` used to select it to how its scikit-learn estimator
    is built for a given contamination rate. A genuinely new detector — one that is not a thin
    wrapper around a scikit-learn estimator — has this one method to implement.
    """

    name: str

    @abstractmethod
    def build(self, contamination: float) -> OutlierMixin:
        """Build a fresh detector configured to flag roughly ``contamination`` of rows.

        Args:
            contamination: The expected proportion of outliers, in ``(0, 0.5]``.

        Returns:
            An unfitted scikit-learn outlier detector exposing ``fit_predict``.
        """
