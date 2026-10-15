"""FastAPI application for the inference service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.common.config import get_settings
from src.common.exceptions import AppError, ModelNotLoadedError, PredictionError, RegistryError
from src.common.logger import get_logger
from src.inference.routes import router
from src.inference.service import InferenceService
from src.registry.manager import ModelRegistryManager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared services and try to load the production model."""

    settings = get_settings()
    registry_manager = ModelRegistryManager(settings)
    inference_service = InferenceService(registry_manager=registry_manager, settings=settings)

    app.state.settings = settings
    app.state.registry_manager = registry_manager
    app.state.inference_service = inference_service

    try:
        inference_service.load_active_model()
    except AppError:
        logger.exception("Inference service started without a loaded production model")

    yield


def create_app() -> FastAPI:
    """Build the FastAPI application."""

    settings = get_settings()
    app = FastAPI(title=settings.project_name, lifespan=lifespan)
    app.include_router(router)

    @app.exception_handler(RegistryError)
    async def registry_error_handler(_: Request, exc: RegistryError) -> JSONResponse:
        logger.exception("Registry error")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(_: Request, exc: ModelNotLoadedError) -> JSONResponse:
        logger.exception("Model is not loaded")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(PredictionError)
    async def prediction_error_handler(_: Request, exc: PredictionError) -> JSONResponse:
        logger.exception("Prediction failed")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc)},
        )

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        logger.exception("Application error")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled application error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()
