"""File-based model registry manager."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.config import Settings, get_settings
from src.common.exceptions import RegistryError
from src.common.logger import get_logger

logger = get_logger(__name__)

MODEL_FILENAME = "model.joblib"
METRICS_FILENAME = "metrics.json"
METADATA_FILENAME = "metadata.json"
ALLOWED_ALIASES = {"production", "latest"}


class ModelRegistryManager:
    """Manage versioned model artifacts in a local file-based registry."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.registry_path = self.settings.model_registry_path
        self.versions_path = self.settings.model_versions_path
        self.production_alias_path = self.settings.production_alias_path
        self.latest_alias_path = self.settings.latest_alias_path

        self.ensure_registry()

    def ensure_registry(self) -> None:
        """Create registry directories if they do not exist."""

        try:
            self.registry_path.mkdir(parents=True, exist_ok=True)
            self.versions_path.mkdir(parents=True, exist_ok=True)
            self.production_alias_path.parent.mkdir(parents=True, exist_ok=True)
            self.latest_alias_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Model registry ensured at %s", self.registry_path)
        except OSError as exc:
            raise RegistryError(f"Failed to initialize registry at {self.registry_path}") from exc

    def get_version_dir(self, version: str) -> Path:
        """Return the directory for a specific model version."""

        return self.versions_path / version

    def register_model(
        self,
        model_path: str | Path,
        metrics: dict[str, Any],
        metadata: dict[str, Any],
        version: str,
    ) -> dict[str, Any]:
        """Register a new model version and update the latest alias."""

        source_model_path = Path(model_path)
        version_dir = self.get_version_dir(version)
        target_model_path = version_dir / MODEL_FILENAME
        metrics_path = version_dir / METRICS_FILENAME
        metadata_path = version_dir / METADATA_FILENAME
        version_dir_created = False

        try:
            if not source_model_path.exists():
                raise RegistryError(f"Model artifact not found: {source_model_path}")

            version_dir.mkdir(parents=True, exist_ok=False)
            version_dir_created = True
            shutil.copy2(source_model_path, target_model_path)
            self._write_json(metrics_path, metrics)
            self._write_json(metadata_path, metadata)

            alias_info = self._build_alias_info(version)
            self._write_json(self.latest_alias_path, alias_info)

            logger.info("Registered model version %s", version)
            return {
                "version": version,
                "version_dir": str(version_dir),
                "model_path": str(target_model_path),
                "metrics_path": str(metrics_path),
                "metadata_path": str(metadata_path),
            }
        except RegistryError:
            if version_dir_created and version_dir.exists():
                shutil.rmtree(version_dir, ignore_errors=True)
            raise
        except FileExistsError as exc:
            raise RegistryError(f"Model version already exists: {version}") from exc
        except OSError as exc:
            if version_dir_created and version_dir.exists():
                shutil.rmtree(version_dir, ignore_errors=True)
            raise RegistryError(f"Failed to register model version: {version}") from exc

    def get_alias_info(self, alias_name: str) -> dict[str, Any] | None:
        """Load alias metadata if the alias exists."""

        alias_path = self._get_alias_path(alias_name)
        if not alias_path.exists():
            return None

        try:
            with alias_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"Failed to read alias file: {alias_path}") from exc

    def set_alias(self, alias_name: str, version: str) -> None:
        """Point an alias to an existing model version."""

        version_dir = self.get_version_dir(version)
        model_path = version_dir / MODEL_FILENAME
        if not model_path.exists():
            raise RegistryError(f"Cannot set alias for missing model version: {version}")

        alias_path = self._get_alias_path(alias_name)
        alias_info = self._build_alias_info(version)

        try:
            self._write_json(alias_path, alias_info)
            logger.info("Alias %s updated to version %s", alias_name, version)
        except OSError as exc:
            raise RegistryError(f"Failed to update alias {alias_name}") from exc

    def get_model_path_by_version(self, version: str) -> Path:
        """Return the artifact path for a model version."""

        model_path = self.get_version_dir(version) / MODEL_FILENAME
        if not model_path.exists():
            raise RegistryError(f"Model not found for version: {version}")
        return model_path

    def get_production_model_path(self) -> Path:
        """Return the current production model path."""

        alias_info = self.get_alias_info("production")
        if alias_info is None:
            raise RegistryError("Production alias is not set")

        version = alias_info.get("version")
        if not version:
            raise RegistryError("Production alias file is invalid")

        return self.get_model_path_by_version(version)

    def load_metrics(self, version: str) -> dict[str, Any] | None:
        """Load metrics for a specific version if present."""

        return self._load_version_json(version, METRICS_FILENAME)

    def load_metadata(self, version: str) -> dict[str, Any] | None:
        """Load metadata for a specific version if present."""

        return self._load_version_json(version, METADATA_FILENAME)

    def _get_alias_path(self, alias_name: str) -> Path:
        if alias_name not in ALLOWED_ALIASES:
            raise RegistryError(f"Unsupported alias: {alias_name}")
        if alias_name == "production":
            return self.production_alias_path
        return self.latest_alias_path

    def _build_alias_info(self, version: str) -> dict[str, Any]:
        model_path = self.get_model_path_by_version(version)
        return {
            "version": version,
            "model_path": str(model_path),
            "updated_at": datetime.now(UTC).isoformat(),
        }

    def _load_version_json(self, version: str, filename: str) -> dict[str, Any] | None:
        target_path = self.get_version_dir(version) / filename
        if not target_path.exists():
            return None

        try:
            with target_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"Failed to read registry file: {target_path}") from exc

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=True, indent=2)
            temp_path.replace(path)
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            raise RegistryError(f"Failed to write registry file: {path}") from exc
