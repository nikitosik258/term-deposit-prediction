"""Shared preprocessing utilities for training and inference."""

import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.common.exceptions import DataValidationError
from src.common.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "y"
ID_COLUMN = None

NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

DERIVED_FEATURES = ["pdays_never"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + DERIVED_FEATURES

# Keep the preprocessing order aligned with the old project.
PREPROCESSOR_NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
    "pdays_never",
]

SCALE_FEATURES = ["balance", "campaign", "previous"]

CAT_ORDERS = [
    [
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
    ],
    ["divorced", "married", "single"],
    ["primary", "secondary", "tertiary", "unknown"],
    ["no", "yes"],
    ["no", "yes"],
    ["no", "yes"],
    ["unknown", "telephone", "cellular"],
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    ["failure", "other", "success", "unknown"],
]


def _to_snake_case(name: str) -> str:
    """Convert a raw column name to snake_case."""

    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized)
    return normalized.strip("_").lower()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the dataframe aligned with the feature contract."""

    cleaned_df = df.copy(deep=True)
    cleaned_df.columns = [_to_snake_case(column) for column in cleaned_df.columns]

    object_columns = cleaned_df.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        cleaned_df[column] = cleaned_df[column].astype("string").str.strip()

    for column in NUMERIC_FEATURES:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")

    if "pdays" in cleaned_df.columns:
        cleaned_df["pdays_never"] = cleaned_df["pdays"].eq(-1).astype("int8")

    if TARGET_COLUMN in cleaned_df.columns:
        cleaned_df[TARGET_COLUMN] = (
            cleaned_df[TARGET_COLUMN].astype("string").str.lower().replace({"yes": 1, "no": 0}).astype("Int64")
        )

    logger.info("Dataframe cleaned with %s rows and %s columns", len(cleaned_df), len(cleaned_df.columns))
    return cleaned_df


def validate_feature_columns(df: pd.DataFrame) -> None:
    """Validate that all required model features are present."""

    missing_columns = sorted(set(ALL_FEATURES) - set(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise DataValidationError(f"Missing required feature columns: {missing}")


def build_preprocessor() -> Pipeline:
    """Build the shared sklearn preprocessor for model training and inference."""

    preprocessor = Pipeline(
        steps=[
            (
                "nan_remover",
                ColumnTransformer(
                    transformers=[
                        (
                            "most_frequent_numeric",
                            SimpleImputer(strategy="most_frequent"),
                            PREPROCESSOR_NUMERIC_FEATURES,
                        ),
                        (
                            "most_frequent_cat",
                            SimpleImputer(strategy="most_frequent"),
                            CATEGORICAL_FEATURES,
                        ),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
            (
                "transformations",
                ColumnTransformer(
                    transformers=[
                        (
                            "encoder",
                            OrdinalEncoder(categories=CAT_ORDERS),
                            CATEGORICAL_FEATURES,
                        ),
                        ("scaler", StandardScaler(), SCALE_FEATURES),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor


def build_training_pipeline(model) -> Pipeline:
    """Build a full training pipeline with preprocessing and estimator."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
