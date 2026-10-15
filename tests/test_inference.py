from fastapi.testclient import TestClient

from src.common.schemas import HealthResponse, ModelInfoResponse, PredictionResponse, ReloadResponse
from src.inference.app import create_app


class FakeInferenceService:
    def load_active_model(self) -> None:
        return None

    def get_health(self) -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=True, model_version="v1")

    def get_model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(model_name="churn_model", model_version="v1")

    def predict(self, payload) -> PredictionResponse:
        return PredictionResponse(
            subscription_probability=0.8123,
            prediction="yes",
            model_version="v1",
        )

    def reload_model(self) -> ReloadResponse:
        return ReloadResponse(status="reloaded", active_model_version="v1")


def create_test_client(monkeypatch) -> TestClient:
    class FakeRegistryManager:
        def __init__(self, settings) -> None:
            self.settings = settings

    monkeypatch.setattr("src.inference.app.ModelRegistryManager", FakeRegistryManager)
    monkeypatch.setattr("src.inference.app.InferenceService", lambda registry_manager, settings: FakeInferenceService())

    app = create_app()
    return TestClient(app)


def valid_payload() -> dict:
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


def test_health_endpoint(monkeypatch) -> None:
    with create_test_client(monkeypatch) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_model_info_endpoint(monkeypatch) -> None:
    with create_test_client(monkeypatch) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_version"] == "v1"


def test_predict_endpoint_with_valid_payload(monkeypatch) -> None:
    with create_test_client(monkeypatch) as client:
        response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    assert response.json()["subscription_probability"] == 0.8123
    assert "churn_probability" not in response.json()
    assert response.json()["prediction"] == "yes"
    assert response.json()["model_version"] == "v1"


def test_predict_endpoint_with_invalid_payload(monkeypatch) -> None:
    invalid_payload = valid_payload()
    invalid_payload.pop("job")

    with create_test_client(monkeypatch) as client:
        response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_openapi_exposes_subscription_probability(monkeypatch) -> None:
    with create_test_client(monkeypatch) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    response_schema = response.json()["components"]["schemas"]["PredictionResponse"]
    properties = response_schema["properties"]
    assert "subscription_probability" in properties
    assert "churn_probability" not in properties
