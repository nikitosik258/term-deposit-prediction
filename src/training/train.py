"""Model training utilities."""

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from src.common.exceptions import TrainingError
from src.common.logger import get_logger
from src.common.preprocessing import build_training_pipeline

logger = get_logger(__name__)


def build_baseline_model(random_state: int) -> LGBMClassifier:
    """Build the baseline term-deposit response classifier."""

    return LGBMClassifier(
        class_weight="balanced",
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=31,
        random_state=random_state,
        verbosity=-1,
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> Pipeline:
    """Train the baseline model wrapped in the shared preprocessing pipeline."""

    logger.info("Starting training on %s rows", len(X_train))
    pipeline = build_training_pipeline(build_baseline_model(random_state))
    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        raise TrainingError("Model training failed") from exc
    logger.info("Training completed successfully")
    return pipeline
