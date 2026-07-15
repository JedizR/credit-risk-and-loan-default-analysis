import numpy as np
import pandas as pd
from sklearn.base import OutlierMixin
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from credit_risk.anomaly.base import AnomalyDetector
from credit_risk.config import CONFIG
from credit_risk.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_risk.pipeline import build_preprocessor


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest — isolates outliers with random splits, cheaply and without a distance."""

    name = "isolation_forest"

    def build(self, contamination: float) -> OutlierMixin:
        return IsolationForest(contamination=contamination, random_state=CONFIG.seed)


class OneClassSVMDetector(AnomalyDetector):
    """One-Class SVM — learns a boundary around the bulk of the data."""

    name = "one_class_svm"

    def build(self, contamination: float) -> OutlierMixin:
        return OneClassSVM(kernel="rbf", nu=contamination, gamma="scale")


DETECTORS: dict[str, AnomalyDetector] = {
    detector.name: detector for detector in (IsolationForestDetector(), OneClassSVMDetector())
}


class UnknownDetectorError(ValueError):
    """Raised when a detector name is not in the registry."""

    def __init__(self, detector_name: str) -> None:
        available = ", ".join(sorted(DETECTORS))
        super().__init__(f"Unknown detector '{detector_name}'. Available detectors: {available}")


def get_detector(name: str) -> AnomalyDetector:
    """Look up a registered detector by name.

    Args:
        name: A registered detector name, e.g. ``"isolation_forest"``.

    Returns:
        The :class:`AnomalyDetector` registered under ``name``.

    Raises:
        UnknownDetectorError: If no detector is registered under ``name``.
    """
    try:
        return DETECTORS[name]
    except KeyError:
        raise UnknownDetectorError(name) from None


def build_detector(detector_name: str, contamination: float | None = None) -> OutlierMixin:
    contamination = contamination or CONFIG.training.outlier_contamination
    return get_detector(detector_name).build(contamination)


def encode_for_detection(frame: pd.DataFrame) -> np.ndarray:
    """Impute, scale and one-hot the mixed features so distance-based detectors can read them."""
    features = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return build_preprocessor().fit_transform(features)


def detect_outliers(
    frame: pd.DataFrame, detector_name: str, contamination: float | None = None
) -> pd.Series:
    detector = build_detector(detector_name, contamination)
    encoded = encode_for_detection(frame)
    flags = detector.fit_predict(encoded) == -1
    return pd.Series(flags, index=frame.index, name=detector_name)


def outlier_flags(frame: pd.DataFrame, contamination: float | None = None) -> pd.DataFrame:
    """One boolean column per detector, aligned to the frame's index."""
    return pd.DataFrame(
        {name: detect_outliers(frame, name, contamination) for name in sorted(DETECTORS)}
    )
