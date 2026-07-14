from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from credit_risk.config import CONFIG
from credit_risk.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_risk.pipeline import build_preprocessor

DETECTOR_BUILDERS: dict[str, Callable[[float], object]] = {
    "isolation_forest": lambda contamination: IsolationForest(
        contamination=contamination,
        random_state=CONFIG.seed,
    ),
    "one_class_svm": lambda contamination: OneClassSVM(
        kernel="rbf",
        nu=contamination,
        gamma="scale",
    ),
}


class UnknownDetectorError(ValueError):
    def __init__(self, detector_name: str) -> None:
        available = ", ".join(sorted(DETECTOR_BUILDERS))
        super().__init__(f"Unknown detector '{detector_name}'. Available detectors: {available}")


def build_detector(detector_name: str, contamination: float | None = None) -> object:
    if detector_name not in DETECTOR_BUILDERS:
        raise UnknownDetectorError(detector_name)
    contamination = contamination or CONFIG.training.outlier_contamination
    return DETECTOR_BUILDERS[detector_name](contamination)


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
        {name: detect_outliers(frame, name, contamination) for name in sorted(DETECTOR_BUILDERS)}
    )
