"""HTTP client helpers used by the Streamlit demonstrator."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import httpx


class PredictionResult(TypedDict):
    """Validated prediction returned by the inference API."""

    subscription_probability: float
    prediction: Literal["yes", "no"]
    model_version: str


class DemoApiError(RuntimeError):
    """An inference API error that can be safely displayed to a user."""


def build_prediction_payload(
    *,
    age: int,
    job: str,
    marital: str,
    education: str,
    default: str,
    balance: int,
    housing: str,
    loan: str,
    contact: str,
    day: int,
    month: str,
    duration: int,
    campaign: int,
    pdays: int,
    previous: int,
    poutcome: str,
) -> dict[str, int | str]:
    """Build a request payload that follows the FastAPI input contract."""

    return {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
    }


def request_prediction(
    payload: dict[str, int | str],
    api_url: str,
    timeout: float = 10.0,
) -> PredictionResult:
    """Send a prediction request and return a validated response."""

    endpoint = f"{api_url.rstrip('/')}/predict"
    try:
        response = httpx.post(endpoint, json=payload, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise DemoApiError("Сервис не ответил вовремя. Попробуйте ещё раз.") from exc
    except httpx.RequestError as exc:
        raise DemoApiError("Не удалось подключиться к сервису прогнозирования.") from exc

    if response.is_error:
        detail = _extract_error_detail(response)
        raise DemoApiError(f"Сервис вернул ошибку {response.status_code}: {detail}")

    try:
        data = response.json()
    except ValueError as exc:
        raise DemoApiError("Сервис вернул ответ в неизвестном формате.") from exc

    return _validate_prediction_response(data)


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return "не удалось прочитать описание ошибки"

    detail = data.get("detail") if isinstance(data, dict) else None
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, list):
        messages = [item.get("msg", str(item)) if isinstance(item, dict) else str(item) for item in detail]
        return "; ".join(messages)
    return "неизвестная ошибка"


def _validate_prediction_response(data: Any) -> PredictionResult:
    if not isinstance(data, dict):
        raise DemoApiError("Сервис вернул некорректную структуру ответа.")

    probability = data.get("subscription_probability")
    prediction = data.get("prediction")
    model_version = data.get("model_version")

    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise DemoApiError("В ответе сервиса отсутствует корректная вероятность.")
    if not 0.0 <= float(probability) <= 1.0:
        raise DemoApiError("Вероятность в ответе сервиса находится вне диапазона от 0 до 1.")
    if prediction not in {"yes", "no"}:
        raise DemoApiError("В ответе сервиса отсутствует корректное предсказание.")
    if not isinstance(model_version, str) or not model_version:
        raise DemoApiError("В ответе сервиса отсутствует версия модели.")

    return {
        "subscription_probability": float(probability),
        "prediction": prediction,
        "model_version": model_version,
    }
