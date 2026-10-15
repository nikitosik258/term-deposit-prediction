import re

import httpx
import numpy as np
import pandas as pd
import pytest

from src.training.deploy import generate_model_version
from src.training.evaluate import evaluate_model
from src.training.run_training import trigger_inference_reload
from src.training.validator import is_model_eligible


class DummyModel:
    def predict_proba(self, X: pd.DataFrame):
        return np.array([[0.2, 0.8], [0.7, 0.3], [0.3, 0.7], [0.8, 0.2]])

    def predict(self, X: pd.DataFrame):
        return [1, 0, 1, 0]


def test_evaluate_model_returns_expected_metrics() -> None:
    model = DummyModel()
    X_val = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y_val = pd.Series([1, 0, 1, 0])

    metrics = evaluate_model(model, X_val, y_val)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}


def test_is_model_eligible_without_current_metrics() -> None:
    decision, reason = is_model_eligible(
        new_metrics={"recall": 0.7, "roc_auc": 0.8, "pr_auc": 0.5},
        current_metrics=None,
        min_recall=0.6,
        min_roc_auc=0.75,
        min_roc_auc_delta=0.0,
        min_pr_auc=0.45,
        min_pr_auc_delta=0.0,
    )

    assert decision is True
    assert "eligible" in reason


def test_is_model_eligible_when_new_model_is_better() -> None:
    decision, _ = is_model_eligible(
        new_metrics={"recall": 0.7, "roc_auc": 0.83, "pr_auc": 0.52},
        current_metrics={"recall": 0.65, "roc_auc": 0.8, "pr_auc": 0.5},
        min_recall=0.6,
        min_roc_auc=0.75,
        min_roc_auc_delta=0.01,
        min_pr_auc=0.45,
        min_pr_auc_delta=0.01,
    )

    assert decision is True


def test_is_model_eligible_when_new_model_is_worse() -> None:
    decision, reason = is_model_eligible(
        new_metrics={"recall": 0.7, "roc_auc": 0.8, "pr_auc": 0.49},
        current_metrics={"recall": 0.65, "roc_auc": 0.82, "pr_auc": 0.5},
        min_recall=0.6,
        min_roc_auc=0.75,
        min_roc_auc_delta=0.0,
        min_pr_auc=0.45,
        min_pr_auc_delta=0.01,
    )

    assert decision is False
    assert "below required baseline" in reason


@pytest.mark.parametrize(
    ("metrics", "expected_metric"),
    [
        ({"recall": 0.59, "roc_auc": 0.8, "pr_auc": 0.5}, "recall"),
        ({"recall": 0.7, "roc_auc": 0.74, "pr_auc": 0.5}, "roc_auc"),
        ({"recall": 0.7, "roc_auc": 0.8, "pr_auc": 0.44}, "pr_auc"),
    ],
)
def test_is_model_eligible_rejects_metrics_below_absolute_thresholds(metrics, expected_metric: str) -> None:
    decision, reason = is_model_eligible(
        new_metrics=metrics,
        current_metrics=None,
        min_recall=0.6,
        min_roc_auc=0.75,
        min_roc_auc_delta=0.0,
        min_pr_auc=0.45,
        min_pr_auc_delta=0.0,
    )

    assert decision is False
    assert expected_metric in reason


def test_is_model_eligible_rejects_pr_auc_below_current_baseline() -> None:
    decision, reason = is_model_eligible(
        new_metrics={"recall": 0.7, "roc_auc": 0.85, "pr_auc": 0.5},
        current_metrics={"recall": 0.65, "roc_auc": 0.8, "pr_auc": 0.5},
        min_recall=0.6,
        min_roc_auc=0.75,
        min_roc_auc_delta=0.01,
        min_pr_auc=0.45,
        min_pr_auc_delta=0.01,
    )

    assert decision is False
    assert "pr_auc" in reason


def test_generate_model_version_returns_expected_format() -> None:
    version = generate_model_version("churn_model")

    assert re.fullmatch(r"churn_model_\d{8}_\d{6}", version) is not None


def test_trigger_inference_reload_skips_when_service_is_unreachable(monkeypatch, caplog) -> None:
    def fake_post(url: str, timeout: float):
        request = httpx.Request("POST", url)
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr("src.training.run_training.httpx.post", fake_post)

    with caplog.at_level("WARNING"):
        trigger_inference_reload("http://127.0.0.1:8000")

    assert "skipping automatic model reload" in caplog.text
