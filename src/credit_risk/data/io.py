import json
from pathlib import Path

import pandas as pd

from credit_risk.config import CONFIG
from credit_risk.data.quality import file_hash, frame_hash, repair_impossible_values
from credit_risk.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    apply_schema_dtypes,
    require_columns,
)

RAW_CSV = CONFIG.paths.raw_csv
PROCESSED_PARQUET = CONFIG.paths.processed_parquet
REGISTRY_JSON = CONFIG.paths.registry_json


def read_frame(path: Path) -> pd.DataFrame:
    """Read a parquet or CSV file into a frame with the project's declared dtypes applied."""
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return apply_schema_dtypes(frame)


def write_parquet(frame: pd.DataFrame, path: Path = PROCESSED_PARQUET) -> None:
    """Write a frame to a typed parquet, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def query(sql: str, source: Path = PROCESSED_PARQUET) -> pd.DataFrame:
    """Run a DuckDB SQL query over a parquet, exposed as the ``applicants`` view.

    DuckDB is a notebook-only dependency imported lazily, so it never enters the runtime image.

    Args:
        sql: A SQL query referencing the ``applicants`` view.
        source: The parquet file to expose as ``applicants``.

    Returns:
        The query result as a DataFrame.

    Raises:
        ModuleNotFoundError: If duckdb is not installed (run ``uv sync --all-groups``).
    """
    try:
        import duckdb
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "duckdb is a notebook-only dependency; run `uv sync --all-groups` to install it"
        ) from error

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
    """Repair the raw CSV into a typed parquet and record its provenance.

    Reads the raw CSV, repairs impossible values, writes the processed parquet, and records the raw
    and processed sha256 hashes plus row/column counts into the data registry.

    Args:
        raw_path: The raw CSV, or None for the configured default.
        parquet_path: Where to write the processed parquet, or None for the default.
        registry_path: Where to write the provenance JSON, or None for the default.

    Returns:
        The repaired, typed frame that was written.
    """
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
    """Load the model's input frame and verify its columns.

    Prefers an explicit ``path``, else the processed parquet, else the raw CSV repaired on the fly.

    Args:
        path: An explicit dataset path, or None to use the prepared parquet / raw CSV.

    Returns:
        A frame containing the feature columns and the target.

    Raises:
        MissingColumnsError: If a required feature or the target column is absent.
    """
    if path is not None:
        frame = read_frame(path)
    elif PROCESSED_PARQUET.exists():
        frame = read_frame(PROCESSED_PARQUET)
    else:
        frame = repair_impossible_values(read_frame(RAW_CSV))
    require_columns(frame, FEATURE_COLUMNS + [TARGET_COLUMN], path or "prepared dataset")
    return frame


def load_features_to_score(path: Path) -> pd.DataFrame:
    """Load new applicants to score, returning only the feature columns.

    Args:
        path: A CSV or parquet of applicants to score.

    Returns:
        The frame restricted to the model's feature columns.

    Raises:
        MissingColumnsError: If a required feature column is absent.
    """
    frame = read_frame(path)
    require_columns(frame, FEATURE_COLUMNS, path)
    return frame[FEATURE_COLUMNS]
