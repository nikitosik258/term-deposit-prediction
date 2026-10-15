"""Inference service logic."""

from __future__ import annotations

from threading import RLock
from typing import Any

import joblib
import pandas as pd

from src.common.config import Settings, get_settings
from src.common.exceptions import ModelNotLoadedError, PredictionError, RegistryError
from src.common.logger import get_logger
from src.common.preprocessing import ALL_FEATURES, TARGET_COLUMN, clean_dataframe, validate_feature_columns
from src.common.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ReloadResponse,
)
from src.registry.manager import ModelRegistryManager

logger = get_logger(__name__)


class InferenceService:
    """Serve predictions from the currently active production model."""

    def __init__(
        self,
        registry_manager: ModelRegistryManager,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry_manager = registry_manager
        self._lock = RLock()
        self._model: Any | None = None
        self._model_name = self.settings.model_name
        self._model_version: str | None = None

    def load_active_model(self) -> None:
        """Load the current production model from the registry."""

        alias_info = self.registry_manager.get_alias_info("production")
        if alias_info is None:
            raise RegistryError("Production alias is not set")

        version = alias_info.get("version")
        if not version:
            raise RegistryError("Production alias file is invalid")

        model_path = self.registry_manager.get_model_path_by_version(version)
        metadata = self.registry_manager.load_metadata(version) or {}
        self._validate_model_metadata(metadata, version)

        try:
            model = joblib.load(model_path)
        except Exception as exc:
            raise RegistryError(f"Failed to load model artifact from {model_path}") from exc

        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            raise RegistryError(f"Loaded artifact for version {version} is not a valid prediction pipeline")

        with self._lock:
            self._model = model
            self._model_version = version
            self._model_name = str(metadata.get("model_name", self.settings.model_name))

        logger.info("Loaded production model version %s from %s", version, model_path)

    def _validate_model_metadata(self, metadata: dict[str, Any], version: str) -> None:
        """Ensure the stored production model matches the current project contract."""

        target_column = metadata.get("target_column")
        feature_columns = metadata.get("feature_columns")

        if target_column != TARGET_COLUMN:
            raise RegistryError(f"Model version {version} is incompatible with current target '{TARGET_COLUMN}'")

        if feature_columns != ALL_FEATURES:
            raise RegistryError(f"Model version {version} is incompatible with current feature contract")

    def predict(self, payload: PredictionRequest) -> PredictionResponse:
        """Predict whether a client will subscribe to a term deposit."""

        with self._lock:
            model = self._model
            model_version = self._model_version

        if model is None or model_version is None:
            raise ModelNotLoadedError("No model is currently loaded")

        inference_df = clean_dataframe(pd.DataFrame([payload.model_dump()]))
        validate_feature_columns(inference_df)

        try:
            subscription_probability = float(model.predict_proba(inference_df)[0][1])
        except Exception as exc:
            raise PredictionError("Failed to generate prediction") from exc

        prediction = "yes" if subscription_probability >= self.settings.subscription_threshold else "no"
        logger.info("Prediction generated with model version %s", model_version)
        return PredictionResponse(
            subscription_probability=round(subscription_probability, 4),
            prediction=prediction,
            model_version=model_version,
        )

    def reload_model(self) -> ReloadResponse:
        """Reload the current production model without dropping the previous one on failure."""

        previous_version = self._model_version
        self.load_active_model()
        logger.info("Model reloaded from version %s to %s", previous_version, self._model_version)
        return ReloadResponse(status="reloaded", active_model_version=self._model_version)

    def get_model_info(self) -> ModelInfoResponse:
        """Return information about the currently loaded model."""

        with self._lock:
            model_name = self._model_name
            model_version = self._model_version

        if model_version is None:
            raise ModelNotLoadedError("No model is currently loaded")

        return ModelInfoResponse(model_name=model_name, model_version=model_version)

    def get_health(self) -> HealthResponse:
        """Return service health based on model availability."""

        with self._lock:
            model_loaded = self._model is not None
            model_version = self._model_version

        return HealthResponse(
            status="ok" if model_loaded else "degraded",
            model_loaded=model_loaded,
            model_version=model_version,
        )
