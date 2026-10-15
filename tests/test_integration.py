from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.common.config import Settings
from src.common.preprocessing import ALL_FEATURES, TARGET_COLUMN, clean_dataframe
from src.inference.app import create_app
from src.registry.manager import ModelRegistryManager
from src.training.data_loader import split_features_target
from src.training.evaluate import evaluate_model
from src.training.train import train_model


class FixedProbabilityModel:
    """Small serializable model artifact for inference integration tests."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(features), self.probability)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), int(self.probability >= 0.5))


def build_test_settings(tmp_path: Path) -> Settings:
    registry_root = tmp_path / "model_registry"
    return Settings(
        MODEL_REGISTRY_PATH=registry_root,
        MODEL_VERSIONS_PATH=registry_root / "versions",
        PRODUCTION_ALIAS_PATH=registry_root / "aliases" / "production.json",
        LATEST_ALIAS_PATH=registry_root / "aliases" / "latest.json",
        MODEL_NAME="churn_model",
        SUBSCRIPTION_THRESHOLD=0.5,
    )


def compatible_metadata() -> dict[str, Any]:
    return {
        "model_name": "churn_model",
        "target_column": TARGET_COLUMN,
        "feature_columns": ALL_FEATURES,
    }


def register_model_artifact(
    manager: ModelRegistryManager,
    tmp_path: Path,
    version: str,
    model: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    artifact_path = tmp_path / f"{version}.joblib"
    joblib.dump(model, artifact_path)
    manager.register_model(
        model_path=artifact_path,
        metrics={"roc_auc": 0.8},
        metadata=metadata or compatible_metadata(),
        version=version,
    )
    manager.set_alias("production", version)


def create_real_client(monkeypatch, settings: Settings) -> TestClient:
    monkeypatch.setattr("src.inference.app.get_settings", lambda: settings)
    return TestClient(create_app())


def valid_payload() -> dict[str, int | str]:
    return {
        "age": 58,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 2143,
        "housing": "yes",
        "loan": "no",
        "contact": "unknown",
        "day": 5,
        "month": "may",
        "duration": 261,
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }


def test_real_inference_service_predicts_through_fastapi(monkeypatch, tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    manager = ModelRegistryManager(settings)
    register_model_artifact(manager, tmp_path, "model_v1", FixedProbabilityModel(0.8123))

    with create_real_client(monkeypatch, settings) as client:
        response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    assert response.json() == {
        "subscription_probability": 0.8123,
        "prediction": "yes",
        "model_version": "model_v1",
    }


def test_reload_model_activates_new_production_version(monkeypatch, tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    manager = ModelRegistryManager(settings)
    register_model_artifact(manager, tmp_path, "model_v1", FixedProbabilityModel(0.2))

    with create_real_client(monkeypatch, settings) as client:
        register_model_artifact(manager, tmp_path, "model_v2", FixedProbabilityModel(0.8))
        reload_response = client.post("/reload-model")
        prediction_response = client.post("/predict", json=valid_payload())

    assert reload_response.status_code == 200
    assert reload_response.json() == {"status": "reloaded", "active_model_version": "model_v2"}
    assert prediction_response.status_code == 200
    assert prediction_response.json()["subscription_probability"] == 0.8
    assert prediction_response.json()["model_version"] == "model_v2"


def test_service_starts_degraded_without_production_alias(monkeypatch, tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)

    with create_real_client(monkeypatch, settings) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "model_loaded": False, "model_version": None}


def test_predict_returns_503_when_model_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)

    with create_real_client(monkeypatch, settings) as client:
        response = client.post("/predict", json=valid_payload())

    assert response.status_code == 503
    assert response.json() == {"detail": "No model is currently loaded"}


def test_service_rejects_missing_model_artifact(monkeypatch, tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    manager = ModelRegistryManager(settings)
    register_model_artifact(manager, tmp_path, "missing_artifact", FixedProbabilityModel(0.7))
    manager.get_model_path_by_version("missing_artifact").unlink()

    with create_real_client(monkeypatch, settings) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_service_rejects_corrupted_model_artifact(monkeypatch, tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    manager = ModelRegistryManager(settings)
    artifact_path = tmp_path / "corrupted.joblib"
    artifact_path.write_bytes(b"not-a-joblib-model")
    manager.register_model(
        model_path=artifact_path,
        metrics={"roc_auc": 0.8},
        metadata=compatible_metadata(),
        version="corrupted_model",
    )
    manager.set_alias("production", "corrupted_model")

    with create_real_client(monkeypatch, settings) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_service_rejects_incompatible_model_metadata(monkeypatch, tmp_path: Path) -> None:
    settings = build_test_settings(tmp_path)
    manager = ModelRegistryManager(settings)
    incompatible_metadata = compatible_metadata()
    incompatible_metadata["feature_columns"] = ["age"]
    register_model_artifact(
        manager,
        tmp_path,
        "incompatible_model",
        FixedProbabilityModel(0.7),
        metadata=incompatible_metadata,
    )

    with create_real_client(monkeypatch, settings) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def build_synthetic_training_data(rows: int = 40) -> pd.DataFrame:
    records = []
    for index in range(rows):
        subscribed = index % 2 == 0
        records.append(
            {
                "age": 25 + index,
                "job": "management" if subscribed else "technician",
                "marital": "married" if subscribed else "single",
                "education": "tertiary" if subscribed else "secondary",
                "default": "no",
                "balance": 2000 + index * 10 if subscribed else -100 - index,
                "housing": "no" if subscribed else "yes",
                "loan": "no" if subscribed else "yes",
                "contact": "cellular" if subscribed else "unknown",
                "day": index % 28 + 1,
                "month": "mar" if subscribed else "may",
                "duration": 400 + index if subscribed else 40 + index,
                "campaign": 1 if subscribed else 4,
                "pdays": 10 if subscribed else -1,
                "previous": 2 if subscribed else 0,
                "poutcome": "success" if subscribed else "unknown",
                "y": "yes" if subscribed else "no",
            }
        )
    return pd.DataFrame.from_records(records)


def test_training_to_registry_to_inference_end_to_end(monkeypatch, tmp_path: Path) -> None:
    dataframe = clean_dataframe(build_synthetic_training_data())
    features, target = split_features_target(dataframe)
    train_features, validation_features = features.iloc[:30], features.iloc[30:]
    train_target, validation_target = target.iloc[:30], target.iloc[30:]

    model = train_model(train_features, train_target, random_state=42)
    metrics = evaluate_model(model, validation_features, validation_target)

    settings = build_test_settings(tmp_path)
    manager = ModelRegistryManager(settings)
    artifact_path = tmp_path / "trained_pipeline.joblib"
    joblib.dump(model, artifact_path)
    manager.register_model(
        model_path=artifact_path,
        metrics=metrics,
        metadata=compatible_metadata(),
        version="trained_pipeline_v1",
    )
    manager.set_alias("production", "trained_pipeline_v1")

    with create_real_client(monkeypatch, settings) as client:
        response = client.post("/predict", json=valid_payload())

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert response.status_code == 200
    assert set(response.json()) == {"subscription_probability", "prediction", "model_version"}
    assert 0.0 <= response.json()["subscription_probability"] <= 1.0
    assert response.json()["prediction"] in {"yes", "no"}
    assert response.json()["model_version"] == "trained_pipeline_v1"
