"""Training data loading utilities."""

from pathlib import Path

import pandas as pd

from src.common.exceptions import DataValidationError, TrainingError
from src.common.logger import get_logger
from src.common.preprocessing import ALL_FEATURES, TARGET_COLUMN, clean_dataframe, validate_feature_columns

logger = get_logger(__name__)


def load_training_data(data_path: str | Path) -> pd.DataFrame:
    """Load and clean training data from a CSV file."""

    csv_path = Path(data_path)
    if not csv_path.exists():
        raise TrainingError(f"Training data file not found: {csv_path}")

    try:
        dataframe = pd.read_csv(csv_path, sep=";")
    except (OSError, ValueError) as exc:
        raise TrainingError(f"Failed to read training data from {csv_path}") from exc

    if dataframe.empty:
        raise TrainingError(f"Training data file is empty: {csv_path}")

    logger.info("Loaded raw training data from %s with shape %s", csv_path, dataframe.shape)

    cleaned_df = clean_dataframe(dataframe)
    if TARGET_COLUMN not in cleaned_df.columns:
        raise TrainingError(f"Target column '{TARGET_COLUMN}' is missing from training data")

    if cleaned_df[TARGET_COLUMN].isna().any():
        raise TrainingError(f"Target column '{TARGET_COLUMN}' contains invalid values")

    logger.info("Prepared cleaned training data with shape %s", cleaned_df.shape)
    return cleaned_df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split cleaned dataframe into features and target."""

    if TARGET_COLUMN not in df.columns:
        raise TrainingError(f"Target column '{TARGET_COLUMN}' is missing from dataframe")

    try:
        validate_feature_columns(df)
    except DataValidationError as exc:
        raise TrainingError(str(exc)) from exc

    features = df.loc[:, ALL_FEATURES].copy()
    target = df.loc[:, TARGET_COLUMN].copy()

    logger.info("Split dataframe into features %s and target %s", features.shape, target.shape)
    return features, target
