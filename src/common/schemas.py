"""Pydantic schemas for inference API and shared contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    """Input payload for predicting a client's response to a bank offer."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    age: int = Field(ge=0)
    job: Literal[
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ]
    marital: Literal["divorced", "married", "single"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: Literal["no", "yes"]
    balance: int
    housing: Literal["no", "yes"]
    loan: Literal["no", "yes"]
    contact: Literal["unknown", "telephone", "cellular"]
    day: int = Field(ge=1, le=31)
    month: Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    duration: int = Field(ge=0)
    campaign: int = Field(ge=0)
    pdays: int
    previous: int = Field(ge=0)
    poutcome: Literal["failure", "other", "success", "unknown"]

    @field_validator(
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "poutcome",
        mode="before",
    )
    @classmethod
    def normalize_string_values(cls, value: str) -> str:
        """Normalize string categorical inputs."""

        if isinstance(value, str):
            return value.strip()
        return value


class PredictionResponse(BaseModel):
    """Term-deposit subscription prediction result."""

    subscription_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability that the client subscribes to a term deposit",
    )
    prediction: Literal["yes", "no"]
    model_version: str = Field(min_length=1)


class HealthResponse(BaseModel):
    """Health check payload."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


class ReloadResponse(BaseModel):
    """Model reload result."""

    status: Literal["reloaded", "failed"]
    active_model_version: str | None = None


class ModelInfoResponse(BaseModel):
    """Loaded model metadata exposed by inference service."""

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
