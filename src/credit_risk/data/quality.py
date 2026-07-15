import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

IMPOSSIBLE_NEGATIVE_FEATURES = ["Income", "LoanAmount"]


def count_impossible_values(frame: pd.DataFrame) -> dict[str, int]:
    """Count the physically impossible negative values in each affected feature."""
    return {
        column: int((frame[column] < 0).sum())
        for column in IMPOSSIBLE_NEGATIVE_FEATURES
        if column in frame.columns
    }


def repair_impossible_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible negatives with NaN so the in-pipeline imputer handles them."""
    repaired = frame.copy()
    for column in IMPOSSIBLE_NEGATIVE_FEATURES:
        if column in repaired.columns:
            repaired.loc[repaired[column] < 0, column] = np.nan
    return repaired


def frame_hash(frame: pd.DataFrame) -> str:
    """Return a sha256 over the frame's row hashes — a content fingerprint for provenance."""
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def file_hash(path: Path) -> str:
    """Return a sha256 of a file's raw bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
