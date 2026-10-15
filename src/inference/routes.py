"""HTTP routes for the inference service."""

from fastapi import APIRouter, Depends, Request

from src.common.schemas import HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse, ReloadResponse
from src.inference.service import InferenceService

router = APIRouter()


def get_inference_service(request: Request) -> InferenceService:
    """Return the shared inference service instance from app state."""

    return request.app.state.inference_service


@router.get("/health", response_model=HealthResponse)
def health(service: InferenceService = Depends(get_inference_service)) -> HealthResponse:
    """Return service health."""

    return service.get_health()


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(service: InferenceService = Depends(get_inference_service)) -> ModelInfoResponse:
    """Return loaded model metadata."""

    return service.get_model_info()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict term-deposit subscription",
    description="Return the probability that a client subscribes to a term deposit after a marketing contact.",
)
def predict(
    payload: PredictionRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Return a term-deposit subscription prediction for one client."""

    return service.predict(payload)


@router.post("/reload-model", response_model=ReloadResponse)
def reload_model(service: InferenceService = Depends(get_inference_service)) -> ReloadResponse:
    """Reload the current production model from the registry."""

    return service.reload_model()
