"""Custom application exceptions."""


class AppError(Exception):
    """Base application error."""


class RegistryError(AppError):
    """Raised for model registry failures."""


class ModelNotLoadedError(AppError):
    """Raised when inference model is not loaded."""


class PredictionError(AppError):
    """Raised when prediction cannot be produced."""


class TrainingError(AppError):
    """Raised when training pipeline fails."""


class DataValidationError(AppError):
    """Raised when input data does not satisfy schema or contract."""
