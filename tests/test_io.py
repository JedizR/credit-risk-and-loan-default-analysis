import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.data import io
from credit_risk.data.io import (
    load_training_data,
    prepare_dataset,
    query,
    read_frame,
    write_parquet,
)
from credit_risk.data.schema import CATEGORICAL_FEATURES, TARGET_COLUMN


def test_read_frame_applies_schema_dtypes(training_csv: Path) -> None:
    frame = read_frame(training_csv)

    assert frame[TARGET_COLUMN].dtype == "int64"
    assert all(str(frame[column].dtype) == "category" for column in CATEGORICAL_FEATURES)


def test_parquet_round_trip_is_dtype_stable(sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    typed = read_frame(_csv(sample_frame, tmp_path))
    parquet_path = tmp_path / "processed" / "applicants.parquet"

    write_parquet(typed, parquet_path)
    restored = read_frame(parquet_path)

    assert restored.dtypes.to_dict() == typed.dtypes.to_dict()
    pd.testing.assert_frame_equal(restored, typed)


def test_read_frame_dispatches_on_suffix(sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    csv_path = _csv(sample_frame, tmp_path)
    parquet_path = tmp_path / "data.parquet"
    write_parquet(read_frame(csv_path), parquet_path)

    assert read_frame(csv_path).equals(read_frame(parquet_path))


def test_prepare_dataset_repairs_and_writes_parquet_and_registry(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "Age": [30],
            "Income": [-5.0],
            "LoanAmount": [1000.0],
            "CreditScore": [700.0],
            "YearsExperience": [3],
            "Gender": ["Male"],
            "Education": ["PhD"],
            "City": ["Chicago"],
            "EmploymentType": ["Salaried"],
            "LoanApproved": [1],
        }
    )
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    parquet_path = tmp_path / "processed" / "applicants.parquet"
    registry_path = tmp_path / "registry.json"

    processed = prepare_dataset(raw_path, parquet_path, registry_path)

    assert processed["Income"].isna().item()
    assert parquet_path.exists()
    registry = json.loads(registry_path.read_text())
    assert registry["n_rows"] == 1
    assert len(registry["raw_sha256"]) == 64


def test_load_training_data_prefers_parquet(monkeypatch, tmp_path: Path) -> None:
    parquet_path = tmp_path / "applicants.parquet"
    write_parquet(read_frame(_csv(_minimal_training_frame(), tmp_path)), parquet_path)
    monkeypatch.setattr(io, "PROCESSED_PARQUET", parquet_path)
    monkeypatch.setattr(io, "RAW_CSV", tmp_path / "does_not_exist.csv")

    frame = load_training_data()

    assert TARGET_COLUMN in frame.columns


def test_load_training_data_falls_back_to_repaired_raw_csv(monkeypatch, tmp_path: Path) -> None:
    raw = _minimal_training_frame()
    raw.loc[0, "Income"] = -1.0
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    monkeypatch.setattr(io, "PROCESSED_PARQUET", tmp_path / "absent.parquet")
    monkeypatch.setattr(io, "RAW_CSV", raw_path)

    frame = load_training_data()

    assert frame.loc[0, "Income"] != frame.loc[0, "Income"]  # repaired negative -> NaN


def test_query_runs_sql_over_parquet(sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    parquet_path = tmp_path / "applicants.parquet"
    write_parquet(read_frame(_csv(sample_frame, tmp_path)), parquet_path)

    result = query("SELECT COUNT(*) AS n FROM applicants", source=parquet_path)

    assert result.loc[0, "n"] == len(sample_frame)


def _csv(frame: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "frame.csv"
    frame.to_csv(path, index=False)
    return path


def _minimal_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [30, 40],
            "Income": [50000.0, 60000.0],
            "LoanAmount": [1000.0, 2000.0],
            "CreditScore": [700.0, 620.0],
            "YearsExperience": [3, 8],
            "Gender": ["Male", "Female"],
            "Education": ["PhD", "Masters"],
            "City": ["Chicago", "Houston"],
            "EmploymentType": ["Salaried", "Self-Employed"],
            "LoanApproved": [1, 0],
        }
    )


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_read_frame_accepts_path_as_string(
    sample_frame: pd.DataFrame, tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / f"data{suffix}"
    if suffix == ".parquet":
        write_parquet(read_frame(_csv(sample_frame, tmp_path)), path)
    else:
        sample_frame.to_csv(path, index=False)

    assert len(read_frame(str(path))) == len(sample_frame)
