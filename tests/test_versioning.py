import json
from pathlib import Path

from credit_risk.config import CONFIG
from credit_risk.pipeline import build_model
from credit_risk.versioning import (
    build_manifest,
    capture_git_sha,
    compute_run_id,
    render_model_card,
    save_run,
    update_registry,
)

INPUTS = {
    "model_name": "lightgbm",
    "params": {"num_leaves": 31},
    "features": ["CreditScore", "Income"],
    "threshold": 0.86,
    "options": {"tune": True},
    "config": {"seed": 42},
    "data": {"raw_sha256": "abc"},
}


def _manifest(**overrides) -> dict:
    base = {
        "model_name": "lightgbm",
        "params": {"num_leaves": 31},
        "features": ["Income", "CreditScore"],
        "threshold": 0.86,
        "metrics": {"average_precision": 0.921, "roc_auc": 0.937},
        "options": {"tune": True, "select_features": False, "remove_outliers": False},
        "created_at": "2026-07-15T10:45:30+00:00",
        "git_sha": "c4407e2",
    }
    return build_manifest(**{**base, **overrides})


def test_run_id_is_deterministic_for_identical_inputs() -> None:
    assert compute_run_id(INPUTS) == compute_run_id(dict(INPUTS))


def test_run_id_ignores_feature_order() -> None:
    reordered = {**INPUTS, "features": list(reversed(INPUTS["features"]))}
    assert compute_run_id({**INPUTS, "features": sorted(INPUTS["features"])}) == compute_run_id(
        {**reordered, "features": sorted(reordered["features"])}
    )


def test_run_id_changes_when_an_input_changes() -> None:
    changed = {**INPUTS, "threshold": 0.5}
    assert compute_run_id(INPUTS) != compute_run_id(changed)


def test_run_id_is_stable_across_git_sha() -> None:
    # The same inputs must map to the same run in the container (git_sha null) and locally.
    local = _manifest(git_sha="c4407e2")
    docker = _manifest(git_sha=None)
    assert local["run_id"] == docker["run_id"]
    assert local["git_sha"] != docker["git_sha"]


def test_git_sha_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("GIT_SHA", "env-sha")
    monkeypatch.setattr("credit_risk.versioning.subprocess.run", _raise)
    assert capture_git_sha() == "env-sha"


def test_git_sha_is_null_when_nothing_is_available(monkeypatch) -> None:
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setattr("credit_risk.versioning.subprocess.run", _raise)
    assert capture_git_sha() is None


def test_manifest_records_provenance_and_metrics() -> None:
    manifest = _manifest()

    assert manifest["metrics"]["average_precision"] == 0.921
    assert manifest["git_sha"] == "c4407e2"
    assert manifest["library_versions"]["credit_risk"]
    assert "lightgbm" in manifest["library_versions"]
    assert manifest["config"]["seed"] == CONFIG.seed


def test_model_card_shows_metrics_and_a_reproduce_command() -> None:
    card = render_model_card(_manifest())

    assert "| average_precision | 0.921 |" in card
    assert "credit-risk train --model-name lightgbm --tune" in card
    assert "c4407e2" in card


def test_registry_appends_without_dropping_and_points_current(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    first = _manifest(threshold=0.5)
    second = _manifest(threshold=0.9)

    update_registry(first, registry)
    update_registry(second, registry)

    data = json.loads(registry.read_text())
    assert set(data["runs"]) == {first["run_id"], second["run_id"]}
    assert data["current"] == second["run_id"]


def test_save_run_writes_model_manifest_card_and_registry(sample_frame, tmp_path: Path) -> None:
    from credit_risk.data.schema import FEATURE_COLUMNS, TARGET_COLUMN

    model = build_model("logistic_regression").fit(
        sample_frame[FEATURE_COLUMNS], sample_frame[TARGET_COLUMN]
    )
    manifest = _manifest()

    record = save_run(
        model,
        manifest,
        models_dir=tmp_path / "models",
        cards_dir=tmp_path / "cards",
        registry_path=tmp_path / "models" / "registry.json",
    )

    assert record.model_path.exists()
    assert record.manifest_path.exists()
    assert record.card_path.exists()
    assert (
        json.loads((tmp_path / "models" / "registry.json").read_text())["current"] == record.run_id
    )
    assert json.loads(record.manifest_path.read_text())["run_id"] == record.run_id


def _raise(*_args, **_kwargs):
    raise FileNotFoundError("git not found")
