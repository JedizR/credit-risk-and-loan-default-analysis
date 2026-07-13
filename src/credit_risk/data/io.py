import json
from pathlib import Path

import duckdb
import pandas as pd

from credit_risk.data.quality import file_hash, frame_hash, repair_impossible_values
from credit_risk.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    apply_schema_dtypes,
    require_columns,
)

DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "loan_risk_prediction_dataset.csv"
PROCESSED_PARQUET = DATA_DIR / "processed" / "applicants.parquet"
REGISTRY_JSON = DATA_DIR / "registry.json"


def read_frame(path: Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return apply_schema_dtypes(frame)


def write_parquet(frame: pd.DataFrame, path: Path = PROCESSED_PARQUET) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def query(sql: str, source: Path = PROCESSED_PARQUET) -> pd.DataFrame:
    connection = duckdb.connect()
    connection.execute(
        f"CREATE VIEW applicants AS SELECT * FROM read_parquet('{Path(source).as_posix()}')"
    )
    return connection.execute(sql).df()


def prepare_dataset(
    raw_path: Path | None = None,
    parquet_path: Path | None = None,
    registry_path: Path | None = None,
) -> pd.DataFrame:
    raw_path = raw_path or RAW_CSV
    parquet_path = parquet_path or PROCESSED_PARQUET
    registry_path = registry_path or REGISTRY_JSON

    processed = repair_impossible_values(read_frame(raw_path))
    write_parquet(processed, parquet_path)

    registry = {
        "raw_csv": Path(raw_path).as_posix(),
        "raw_sha256": file_hash(raw_path),
        "processed_parquet": Path(parquet_path).as_posix(),
        "processed_sha256": frame_hash(processed),
        "n_rows": int(len(processed)),
        "n_columns": int(processed.shape[1]),
    }
    Path(registry_path).write_text(json.dumps(registry, indent=2) + "\n")
    return processed


def load_training_data(path: Path | None = None) -> pd.DataFrame:
    if path is not None:
        frame = read_frame(path)
    elif PROCESSED_PARQUET.exists():
        frame = read_frame(PROCESSED_PARQUET)
    else:
        frame = repair_impossible_values(read_frame(RAW_CSV))
    require_columns(frame, FEATURE_COLUMNS + [TARGET_COLUMN], path or "prepared dataset")
    return frame


def load_features_to_score(path: Path) -> pd.DataFrame:
    frame = read_frame(path)
    require_columns(frame, FEATURE_COLUMNS, path)
    return frame[FEATURE_COLUMNS]
