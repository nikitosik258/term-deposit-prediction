"""Model evaluation utilities."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.common.exceptions import TrainingError
from src.common.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
    """Compute validation metrics for a trained model."""

    try:
        probabilities = model.predict_proba(X_val)[:, 1]
        predictions = model.predict(X_val)
    except Exception as exc:
        raise TrainingError("Failed to generate validation predictions") from exc

    try:
        metrics = {
            "accuracy": round(float(accuracy_score(y_val, predictions)), 4),
            "precision": round(float(precision_score(y_val, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_val, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_val, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_val, probabilities)), 4),
            "pr_auc": round(float(average_precision_score(y_val, probabilities)), 4),
        }
    except Exception as exc:
        raise TrainingError("Failed to compute validation metrics") from exc

    logger.info("Validation metrics: %s", metrics)
    return metrics
