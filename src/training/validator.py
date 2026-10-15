"""Promotion validation rules for trained models."""

from typing import Any

from src.common.logger import get_logger

logger = get_logger(__name__)


def is_model_eligible(
    new_metrics: dict[str, Any],
    current_metrics: dict[str, Any] | None,
    min_recall: float,
    min_roc_auc: float,
    min_roc_auc_delta: float,
    min_pr_auc: float,
    min_pr_auc_delta: float,
) -> tuple[bool, str]:
    """Decide whether a new model can be promoted to production."""

    new_recall = float(new_metrics.get("recall", 0.0))
    new_roc_auc = float(new_metrics.get("roc_auc", 0.0))
    new_pr_auc = float(new_metrics.get("pr_auc", 0.0))

    if new_recall < min_recall:
        reason = f"Model recall {new_recall:.4f} is below minimum threshold {min_recall:.4f}"
        logger.info("Model promotion rejected: %s", reason)
        return False, reason

    if new_roc_auc < min_roc_auc:
        reason = f"Model roc_auc {new_roc_auc:.4f} is below minimum threshold {min_roc_auc:.4f}"
        logger.info("Model promotion rejected: %s", reason)
        return False, reason

    if new_pr_auc < min_pr_auc:
        reason = f"Model pr_auc {new_pr_auc:.4f} is below minimum threshold {min_pr_auc:.4f}"
        logger.info("Model promotion rejected: %s", reason)
        return False, reason

    if current_metrics is not None:
        current_roc_auc = float(current_metrics.get("roc_auc", 0.0))
        required_roc_auc = current_roc_auc + min_roc_auc_delta
        if new_roc_auc < required_roc_auc:
            reason = f"Model roc_auc {new_roc_auc:.4f} is below required baseline {required_roc_auc:.4f}"
            logger.info("Model promotion rejected: %s", reason)
            return False, reason

        current_pr_auc = float(current_metrics.get("pr_auc", 0.0))
        required_pr_auc = current_pr_auc + min_pr_auc_delta
        if new_pr_auc < required_pr_auc:
            reason = f"Model pr_auc {new_pr_auc:.4f} is below required baseline {required_pr_auc:.4f}"
            logger.info("Model promotion rejected: %s", reason)
            return False, reason

    reason = "Model passed validation and is eligible for promotion"
    logger.info("Model promotion approved: %s", reason)
    return True, reason
