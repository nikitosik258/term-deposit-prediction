import httpx
import pytest

from src.demo.client import DemoApiError, build_prediction_payload, request_prediction


def valid_payload() -> dict[str, int | str]:
    return build_prediction_payload(
        age=58,
        job="management",
        marital="married",
        education="tertiary",
        default="no",
        balance=2143,
        housing="yes",
        loan="no",
        contact="unknown",
        day=5,
        month="may",
        duration=261,
        campaign=1,
        pdays=-1,
        previous=0,
        poutcome="unknown",
    )


def make_response(status_code: int, **kwargs) -> httpx.Response:
    request = httpx.Request("POST", "http://api:8000/predict")
    return httpx.Response(status_code, request=request, **kwargs)


def test_build_prediction_payload_contains_all_api_fields() -> None:
    payload = valid_payload()

    assert len(payload) == 16
    assert payload["age"] == 58
    assert payload["pdays"] == -1
    assert payload["poutcome"] == "unknown"


def test_request_prediction_returns_validated_result(monkeypatch) -> None:
    response = make_response(
        200,
        json={
            "subscription_probability": 0.8123,
            "prediction": "yes",
            "model_version": "v1",
        },
    )
    monkeypatch.setattr("src.demo.client.httpx.post", lambda *args, **kwargs: response)

    result = request_prediction(valid_payload(), "http://api:8000/")

    assert result["subscription_probability"] == 0.8123
    assert result["prediction"] == "yes"


def test_request_prediction_handles_timeout(monkeypatch) -> None:
    def raise_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timeout", request=httpx.Request("POST", "http://api:8000/predict"))

    monkeypatch.setattr("src.demo.client.httpx.post", raise_timeout)

    with pytest.raises(DemoApiError, match="не ответил вовремя"):
        request_prediction(valid_payload(), "http://api:8000")


def test_request_prediction_handles_unavailable_service(monkeypatch) -> None:
    def raise_connection_error(*args, **kwargs):
        raise httpx.ConnectError("unavailable", request=httpx.Request("POST", "http://api:8000/predict"))

    monkeypatch.setattr("src.demo.client.httpx.post", raise_connection_error)

    with pytest.raises(DemoApiError, match="Не удалось подключиться"):
        request_prediction(valid_payload(), "http://api:8000")


@pytest.mark.parametrize("status_code", [422, 500])
def test_request_prediction_handles_http_errors(monkeypatch, status_code: int) -> None:
    response = make_response(status_code, json={"detail": "API failure"})
    monkeypatch.setattr("src.demo.client.httpx.post", lambda *args, **kwargs: response)

    with pytest.raises(DemoApiError, match=rf"ошибку {status_code}: API failure"):
        request_prediction(valid_payload(), "http://api:8000")


def test_request_prediction_handles_invalid_json(monkeypatch) -> None:
    response = make_response(200, content=b"not-json")
    monkeypatch.setattr("src.demo.client.httpx.post", lambda *args, **kwargs: response)

    with pytest.raises(DemoApiError, match="неизвестном формате"):
        request_prediction(valid_payload(), "http://api:8000")


def test_request_prediction_rejects_invalid_response_contract(monkeypatch) -> None:
    response = make_response(200, json={"prediction": "yes", "model_version": "v1"})
    monkeypatch.setattr("src.demo.client.httpx.post", lambda *args, **kwargs: response)

    with pytest.raises(DemoApiError, match="корректная вероятность"):
        request_prediction(valid_payload(), "http://api:8000")


def test_request_prediction_formats_validation_errors(monkeypatch) -> None:
    response = make_response(422, json={"detail": [{"msg": "Field required"}, {"msg": "Invalid value"}]})
    monkeypatch.setattr("src.demo.client.httpx.post", lambda *args, **kwargs: response)

    with pytest.raises(DemoApiError, match="Field required; Invalid value"):
        request_prediction(valid_payload(), "http://api:8000")


def test_request_prediction_handles_non_json_http_error(monkeypatch) -> None:
    response = make_response(500, content=b"broken-response")
    monkeypatch.setattr("src.demo.client.httpx.post", lambda *args, **kwargs: response)

    with pytest.raises(DemoApiError, match="не удалось прочитать описание ошибки"):
        request_prediction(valid_payload(), "http://api:8000")


@pytest.mark.parametrize(
    ("response_data", "message"),
    [
        (
            {"subscription_probability": 1.5, "prediction": "yes", "model_version": "v1"},
            "вне диапазона",
        ),
        (
            {"subscription_probability": 0.5, "prediction": "maybe", "model_version": "v1"},
            "корректное предсказание",
        ),
        (
            {"subscription_probability": 0.5, "prediction": "yes", "model_version": ""},
            "версия модели",
        ),
    ],
)
def test_request_prediction_rejects_invalid_response_values(monkeypatch, response_data: dict, message: str) -> None:
    response = make_response(200, json=response_data)
    monkeypatch.setattr("src.demo.client.httpx.post", lambda *args, **kwargs: response)

    with pytest.raises(DemoApiError, match=message):
        request_prediction(valid_payload(), "http://api:8000")
