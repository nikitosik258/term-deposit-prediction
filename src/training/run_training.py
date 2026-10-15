"""Entrypoint for the on-demand training service."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from typing import Any

import httpx
from sklearn.model_selection import train_test_split

from src.common.config import get_settings
from src.common.exceptions import RegistryError, TrainingError
from src.common.logger import get_logger
from src.common.preprocessing import ALL_FEATURES, TARGET_COLUMN
from src.registry.manager import ModelRegistryManager
from src.training.data_loader import load_training_data, split_features_target
from src.training.deploy import deploy_model
from src.training.evaluate import evaluate_model
from src.training.train import train_model
from src.training.validator import is_model_eligible

logger = get_logger(__name__)


def trigger_inference_reload(inference_url: str) -> None:
    """Notify inference service to reload the active production model."""

    reload_url = f"{inference_url.rstrip('/')}/reload-model"

    try:
        response = httpx.post(reload_url, timeout=10.0)
        response.raise_for_status()
        logger.info("Inference model reload triggered successfully via %s", reload_url)
    except httpx.ConnectError:
        logger.warning(
            "Inference service is not reachable at %s; skipping automatic model reload",
            reload_url,
        )
    except httpx.HTTPError:
        logger.exception("Failed to trigger inference model reload via %s", reload_url)


def _load_current_production_metrics(
    registry_manager: ModelRegistryManager,
) -> tuple[str | None, dict[str, Any] | None]:
    alias_info = registry_manager.get_alias_info("production")
    if alias_info is None:
        logger.info("No production model is currently registered")
        return None, None

    version = alias_info.get("version")
    if not version:
        raise RegistryError("Production alias does not contain a model version")

    metadata = registry_manager.load_metadata(version) or {}
    if metadata.get("target_column") != TARGET_COLUMN or metadata.get("feature_columns") != ALL_FEATURES:
        logger.info(
            "Current production model version %s is incompatible with the current contract and will be ignored",
            version,
        )
        return None, None

    metrics = registry_manager.load_metrics(version)
    logger.info("Loaded current production metrics for version %s", version)
    return version, metrics


def main() -> int:
    """Run the full training job once and exit."""

    settings = get_settings()

    try:
        registry_manager = ModelRegistryManager(settings)
        training_df = load_training_data(settings.raw_data_path)
        features, target = split_features_target(training_df)

        if target.nunique() < 2:
            raise TrainingError("Training target must contain at least two classes")

        X_train, X_val, y_train, y_val = train_test_split(
            features,
            target,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=target,
        )

        model = train_model(X_train, y_train, random_state=settings.random_state)
        new_metrics = evaluate_model(model, X_val, y_val)

        current_version, current_metrics = _load_current_production_metrics(registry_manager)
        is_eligible, reason = is_model_eligible(
            new_metrics=new_metrics,
            current_metrics=current_metrics,
            min_recall=settings.min_recall,
            min_roc_auc=settings.min_roc_auc,
            min_roc_auc_delta=settings.min_roc_auc_delta,
            min_pr_auc=settings.min_pr_auc,
            min_pr_auc_delta=settings.min_pr_auc_delta,
        )

        if not is_eligible:
            logger.info("Training finished without promotion: %s", reason)
            return 0

        metadata = {
            "algorithm": "LGBMClassifier",
            "target_column": TARGET_COLUMN,
            "feature_columns": ALL_FEATURES,
            "source_data_path": str(settings.raw_data_path),
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "evaluated_at": datetime.now(UTC).isoformat(),
            "previous_production_version": current_version,
            "promotion_reason": reason,
        }
        deployment_info = deploy_model(
            registry_manager=registry_manager,
            model=model,
            metrics=new_metrics,
            metadata=metadata,
            model_name=settings.model_name,
        )

        logger.info("New production model deployed: %s", deployment_info["version"])
        trigger_inference_reload(settings.inference_url)
        return 0
    except Exception as exc:
        logger.exception("Training job failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
