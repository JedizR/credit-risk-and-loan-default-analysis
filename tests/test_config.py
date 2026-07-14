from pathlib import Path

from credit_risk.config import CONFIG, Config, load_config


def test_defaults_are_loaded_when_no_config_file_exists(tmp_path: Path) -> None:
    config = load_config(tmp_path / "absent.toml")

    assert config == Config()
    assert config.seed == 42


def test_toml_overrides_only_the_keys_it_names(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "seed = 7\n\n[training]\nmodel_name = 'random_forest'\noptuna_trials = 3\n"
    )

    config = load_config(config_file)

    assert config.seed == 7
    assert config.training.model_name == "random_forest"
    assert config.training.optuna_trials == 3
    assert config.training.holdout_fraction == Config().training.holdout_fraction


def test_paths_from_toml_become_path_objects(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("[paths]\nmodel = 'artifacts/best.joblib'\n")

    config = load_config(config_file)

    assert config.paths.model == Path("artifacts/best.joblib")


def test_with_training_returns_a_new_config_without_mutating() -> None:
    tuned = CONFIG.with_training(optuna_trials=99)

    assert tuned.training.optuna_trials == 99
    assert CONFIG.training.optuna_trials != 99
    assert tuned.seed == CONFIG.seed


def test_config_serializes_for_run_provenance() -> None:
    snapshot = CONFIG.as_dict()

    assert snapshot["seed"] == CONFIG.seed
    assert snapshot["training"]["model_name"] == CONFIG.training.model_name


def test_single_seed_drives_every_module() -> None:
    from credit_risk import pipeline
    from credit_risk.eda import profile

    model = pipeline.build_model("random_forest")

    assert model.named_steps["classifier"].random_state == CONFIG.seed
    assert CONFIG.thresholds.signal_mutual_info == profile.SIGNAL_MUTUAL_INFO
