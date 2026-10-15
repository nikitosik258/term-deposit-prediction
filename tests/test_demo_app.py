from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.demo.client import DemoApiError


def build_app() -> AppTest:
    app_path = Path(__file__).parents[1] / "src" / "demo" / "app.py"
    return AppTest.from_file(app_path, default_timeout=15)


def test_demo_renders_all_prediction_fields() -> None:
    app = build_app().run()

    assert not app.exception
    assert app.title[0].value == "Прогноз отклика на банковское предложение"
    assert len(app.number_input) == 7
    assert len(app.selectbox) == 9
    assert len(app.button) == 1


def test_demo_displays_successful_prediction() -> None:
    prediction = {
        "subscription_probability": 0.8123,
        "prediction": "yes",
        "model_version": "model_v1",
    }

    with patch("src.demo.client.request_prediction", return_value=prediction):
        app = build_app().run()
        app.button[0].click().run()

    assert not app.exception
    assert app.metric[0].value == "81.23%"
    assert "откликнется" in app.success[0].value
    assert "model_v1" in app.caption[-1].value


def test_demo_displays_negative_prediction() -> None:
    prediction = {
        "subscription_probability": 0.2,
        "prediction": "no",
        "model_version": "model_v2",
    }

    with patch("src.demo.client.request_prediction", return_value=prediction):
        app = build_app().run()
        app.button[0].click().run()

    assert not app.exception
    assert app.metric[0].value == "20.00%"
    assert "не откликнется" in app.info[0].value


def test_demo_displays_api_error_without_crashing() -> None:
    with patch("src.demo.client.request_prediction", side_effect=DemoApiError("API недоступен")):
        app = build_app().run()
        app.button[0].click().run()

    assert not app.exception
    assert app.error[0].value == "API недоступен"
