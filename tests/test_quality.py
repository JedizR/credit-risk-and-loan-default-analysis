import numpy as np
import pandas as pd

from credit_risk.data.quality import (
    count_impossible_values,
    file_hash,
    frame_hash,
    repair_impossible_values,
)


def test_count_impossible_values_counts_only_negatives() -> None:
    frame = pd.DataFrame({"Income": [-1.0, 5.0, -3.0], "LoanAmount": [10.0, -2.0, 4.0]})

    assert count_impossible_values(frame) == {"Income": 2, "LoanAmount": 1}


def test_repair_replaces_negatives_with_nan_and_keeps_valid_values() -> None:
    frame = pd.DataFrame({"Income": [-1.0, 5.0], "LoanAmount": [10.0, -2.0]})

    repaired = repair_impossible_values(frame)

    assert repaired["Income"].isna().tolist() == [True, False]
    assert repaired["LoanAmount"].isna().tolist() == [False, True]
    assert repaired.loc[1, "Income"] == 5.0


def test_repair_leaves_original_frame_untouched() -> None:
    frame = pd.DataFrame({"Income": [-1.0]})

    repair_impossible_values(frame)

    assert frame.loc[0, "Income"] == -1.0


def test_frame_hash_is_deterministic_and_content_sensitive() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    changed = pd.DataFrame({"a": [1, 3], "b": ["x", "y"]})

    assert frame_hash(frame) == frame_hash(frame.copy())
    assert frame_hash(frame) != frame_hash(changed)


def test_file_hash_matches_written_bytes(tmp_path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"credit-risk")

    import hashlib

    assert file_hash(path) == hashlib.sha256(b"credit-risk").hexdigest()


def test_repair_handles_frame_without_target_or_extra_columns() -> None:
    frame = pd.DataFrame({"Age": [25], "Income": [np.nan]})

    repaired = repair_impossible_values(frame)

    assert repaired["Income"].isna().item()
