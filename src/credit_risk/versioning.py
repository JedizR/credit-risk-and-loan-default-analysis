import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from credit_risk import __version__
from credit_risk.config import CONFIG

RUN_ID_LENGTH = 12
TRACKED_LIBRARIES = ["lightgbm", "scikit-learn", "shap", "optuna", "numpy", "pandas"]


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    manifest: dict
    model_path: Path
    manifest_path: Path
    card_path: Path


def compute_run_id(inputs: dict) -> str:
    """A content hash of the inputs that define the model.

    Deterministic on purpose: the same data, config and parameters produce the same id on any
    machine at any time, so reproducibility can be asserted. `created_at`, `git_sha` and library
    versions are recorded but excluded here — `git_sha` is null inside the container, and folding
    it in would give the same inputs a different id depending on where they ran.
    """
    canonical = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:RUN_ID_LENGTH]


def capture_git_sha() -> str | None:
    """Best-effort code version: git if a checkout is present, else the GIT_SHA env, else null."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return completed.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return os.environ.get("GIT_SHA") or None


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "credit_risk": __version__}
    for library in TRACKED_LIBRARIES:
        try:
            versions[library] = version(library)
        except PackageNotFoundError:
            continue
    return versions


def _data_provenance(registry_path: Path | None = None) -> dict:
    """The data-stage hashes this model was trained on, linking the two provenance chains."""
    registry_path = registry_path or CONFIG.paths.registry_json
    if not Path(registry_path).exists():
        return {}
    registry = json.loads(Path(registry_path).read_text())
    return {key: registry[key] for key in ("raw_sha256", "processed_sha256") if key in registry}


def build_manifest(
    model_name: str,
    params: dict,
    features: list[str],
    threshold: float,
    metrics: dict,
    options: dict,
    created_at: str | None = None,
    git_sha: str | None = None,
    data_registry_path: Path | None = None,
) -> dict:
    inputs = {
        "model_name": model_name,
        "params": params,
        "features": sorted(features),
        "threshold": threshold,
        "options": options,
        "config": CONFIG.as_dict(),
        "data": _data_provenance(data_registry_path),
    }
    return {
        "run_id": compute_run_id(inputs),
        "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha if git_sha is not None else capture_git_sha(),
        "library_versions": library_versions(),
        "metrics": metrics,
        **inputs,
    }


def render_model_card(manifest: dict) -> str:
    metric_rows = "\n".join(f"| {name} | {value} |" for name, value in manifest["metrics"].items())
    library_rows = "\n".join(
        f"- {name}: {ver}" for name, ver in manifest["library_versions"].items()
    )
    return f"""# Model card — run `{manifest["run_id"]}`

Loan-approval classifier. This card is generated from the run manifest; it is a record of what
produced the model and how to reproduce it.

- **Model:** {manifest["model_name"]}
- **Created:** {manifest["created_at"]}
- **Code version:** {manifest["git_sha"] or "unknown (no git checkout)"}
- **Decision threshold:** {manifest["threshold"]}

## Metrics (holdout)

| metric | value |
| --- | --- |
{metric_rows}

## Data provenance

{_render_data(manifest["data"])}

## Configuration and features

- **Run options:** {json.dumps(manifest["options"])}
- **Tuned parameters:** {json.dumps(manifest["params"])}
- **Features ({len(manifest["features"])}):** {", ".join(manifest["features"])}

## Reproduce

```bash
credit-risk train --model-name {manifest["model_name"]}{_reproduce_flags(manifest["options"])}
```

Identical data, config and parameters reproduce run id `{manifest["run_id"]}`.

## Library versions

{library_rows}
"""


def _render_data(data: dict) -> str:
    if not data:
        return "No data registry was found for this run."
    return "\n".join(f"- **{key}:** `{value}`" for key, value in data.items())


def _reproduce_flags(options: dict) -> str:
    flags = {
        "tune": "--tune",
        "select_features": "--select-features",
        "remove_outliers": "--remove-outliers",
    }
    return "".join(f" {flag}" for key, flag in flags.items() if options.get(key))


def update_registry(manifest: dict, registry_path: Path | None = None) -> None:
    """Append the run to the index and point `current` at it, without dropping earlier runs."""
    registry_path = Path(registry_path or CONFIG.paths.model_registry)
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"runs": {}}

    registry["runs"][manifest["run_id"]] = {
        "created_at": manifest["created_at"],
        "model_name": manifest["model_name"],
        "average_precision": manifest["metrics"].get("average_precision"),
        "roc_auc": manifest["metrics"].get("roc_auc"),
        "feature_count": len(manifest["features"]),
        "git_sha": manifest["git_sha"],
    }
    registry["current"] = manifest["run_id"]

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")


def save_run(
    model: Pipeline,
    manifest: dict,
    models_dir: Path | None = None,
    cards_dir: Path | None = None,
    registry_path: Path | None = None,
) -> RunRecord:
    """Persist the versioned model, its manifest, its model card, and the registry entry."""
    models_dir = Path(models_dir or CONFIG.paths.models_dir)
    cards_dir = Path(cards_dir or CONFIG.paths.model_cards)
    run_id = manifest["run_id"]

    models_dir.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{run_id}.joblib"
    manifest_path = models_dir / f"{run_id}.meta.json"
    card_path = cards_dir / f"{run_id}.md"

    joblib.dump(model, model_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    card_path.write_text(render_model_card(manifest))
    update_registry(manifest, registry_path or models_dir / "registry.json")

    return RunRecord(run_id, manifest, model_path, manifest_path, card_path)
