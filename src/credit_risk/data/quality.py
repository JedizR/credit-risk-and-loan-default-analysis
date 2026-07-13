import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

IMPOSSIBLE_NEGATIVE_FEATURES = ["Income", "LoanAmount"]


def count_impossible_values(frame: pd.DataFrame) -> dict[str, int]:
    return {
        column: int((frame[column] < 0).sum())
        for column in IMPOSSIBLE_NEGATIVE_FEATURES
        if column in frame.columns
    }


def repair_impossible_values(frame: pd.DataFrame) -> pd.DataFrame:
    repaired = frame.copy()
    for column in IMPOSSIBLE_NEGATIVE_FEATURES:
        if column in repaired.columns:
            repaired.loc[repaired[column] < 0, column] = np.nan
    return repaired


def frame_hash(frame: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
