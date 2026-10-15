from pathlib import Path

from src.common.config import Settings
from src.registry.manager import ModelRegistryManager


def build_test_settings(tmp_path: Path) -> Settings:
    registry_root = tmp_path / "model_registry"
    return Settings(
        MODEL_REGISTRY_PATH=registry_root,
        MODEL_VERSIONS_PATH=registry_root / "versions",
        PRODUCTION_ALIAS_PATH=registry_root / "aliases" / "production.json",
        LATEST_ALIAS_PATH=registry_root / "aliases" / "latest.json",
    )


def test_ensure_registry_creates_expected_structure(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    manager = ModelRegistryManager(settings)

    assert manager.registry_path.exists()
    assert manager.versions_path.exists()
    assert manager.production_alias_path.parent.exists()
    assert manager.latest_alias_path.parent.exists()


def test_register_model_saves_version_artifacts_and_latest_alias(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    manager = ModelRegistryManager(settings)
    source_model = tmp_path / "source_model.joblib"
    source_model.write_bytes(b"fake-model")

    result = manager.register_model(
        model_path=source_model,
        metrics={"roc_auc": 0.81},
        metadata={"model_name": "churn_model"},
        version="churn_model_v1",
    )

    version_dir = manager.get_version_dir("churn_model_v1")
    assert Path(result["model_path"]).exists()
    assert (version_dir / "metrics.json").exists()
    assert (version_dir / "metadata.json").exists()
    assert manager.get_alias_info("latest")["version"] == "churn_model_v1"


def test_set_alias_updates_production_alias(tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    manager = ModelRegistryManager(settings)
    source_model = tmp_path / "source_model.joblib"
    source_model.write_bytes(b"fake-model")

    manager.register_model(
        model_path=source_model,
        metrics={"roc_auc": 0.81},
        metadata={"model_name": "churn_model"},
        version="churn_model_v1",
    )
    manager.set_alias("production", "churn_model_v1")

    alias_info = manager.get_alias_info("production")
    assert alias_info is not None
    assert alias_info["version"] == "churn_model_v1"
    assert manager.get_production_model_path().name == "model.joblib"
