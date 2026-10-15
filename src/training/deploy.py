"""Model deployment helpers for the training job."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from src.common.exceptions import RegistryError, TrainingError
from src.common.logger import get_logger
from src.registry.manager import ModelRegistryManager

logger = get_logger(__name__)


def generate_model_version(model_name: str) -> str:
    """Generate a unique model version name."""

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}"


def save_temp_model(model: Pipeline, output_path: Path) -> Path:
    """Persist a trained model pipeline to a temporary location."""

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
    except OSError as exc:
        raise TrainingError(f"Failed to save model artifact to {output_path}") from exc

    logger.info("Temporary model artifact saved to %s", output_path)
    return output_path


def deploy_model(
    registry_manager: ModelRegistryManager,
    model: Pipeline,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Save, register, and promote a new model version."""

    version = generate_model_version(model_name)
    deployment_metadata = {
        **metadata,
        "model_name": model_name,
        "version": version,
        "registered_at": datetime.now(UTC).isoformat(),
    }

    try:
        with TemporaryDirectory() as temp_dir:
            temp_model_path = save_temp_model(model, Path(temp_dir) / "model.joblib")
            registry_info = registry_manager.register_model(
                model_path=temp_model_path,
                metrics=metrics,
                metadata=deployment_metadata,
                version=version,
            )

        registry_manager.set_alias("production", version)
    except (RegistryError, TrainingError):
        raise
    except OSError as exc:
        raise TrainingError(f"Failed to deploy model version {version}") from exc

    logger.info("Model version %s promoted to production", version)
    return {**registry_info, "version": version}
