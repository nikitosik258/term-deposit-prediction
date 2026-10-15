"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized project settings."""

    project_name: str = Field(default="bank-marketing-response-service", alias="PROJECT_NAME")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_registry_path: Path = Field(default=Path("/app/model_registry"), alias="MODEL_REGISTRY_PATH")
    model_versions_path: Path = Field(default=Path("/app/model_registry/versions"), alias="MODEL_VERSIONS_PATH")
    production_alias_path: Path = Field(
        default=Path("/app/model_registry/aliases/production.json"),
        alias="PRODUCTION_ALIAS_PATH",
    )
    latest_alias_path: Path = Field(
        default=Path("/app/model_registry/aliases/latest.json"),
        alias="LATEST_ALIAS_PATH",
    )
    raw_data_path: Path = Field(
        default=Path("/app/data/raw/data.csv"),
        alias="RAW_DATA_PATH",
    )

    inference_host: str = Field(default="0.0.0.0", alias="INFERENCE_HOST")
    inference_port: int = Field(default=8000, alias="INFERENCE_PORT")
    inference_url: str = Field(default="http://inference:8000", alias="INFERENCE_URL")

    model_name: str = Field(default="churn_model", alias="MODEL_NAME")
    subscription_threshold: float = Field(
        default=0.5,
        validation_alias=AliasChoices("SUBSCRIPTION_THRESHOLD", "CHURN_THRESHOLD"),
    )
    min_recall: float = Field(default=0.6, alias="MIN_RECALL")
    min_roc_auc: float = Field(default=0.75, alias="MIN_ROC_AUC")
    min_roc_auc_delta: float = Field(default=0.0, alias="MIN_ROC_AUC_DELTA")
    min_pr_auc: float = Field(default=0.45, alias="MIN_PR_AUC")
    min_pr_auc_delta: float = Field(default=0.0, alias="MIN_PR_AUC_DELTA")
    random_state: int = Field(default=42, alias="RANDOM_STATE")
    test_size: float = Field(default=0.2, alias="TEST_SIZE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
