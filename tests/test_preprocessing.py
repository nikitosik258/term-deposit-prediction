import pandas as pd
import pytest

from src.common.exceptions import DataValidationError
from src.common.preprocessing import clean_dataframe, validate_feature_columns


def test_clean_dataframe_converts_numeric_columns_and_creates_pdays_never() -> None:
    dataframe = pd.DataFrame(
        {
            "age": ["58", "44"],
            "balance": ["2143", "29"],
            "day": ["5", "5"],
            "duration": ["261", "151"],
            "campaign": ["1", "1"],
            "pdays": ["-1", "12"],
            "previous": ["0", "1"],
        }
    )

    cleaned = clean_dataframe(dataframe)

    assert cleaned["balance"].tolist() == [2143, 29]
    assert cleaned["pdays_never"].tolist() == [1, 0]
    assert cleaned["age"].dtype.kind in {"i", "f"}


def test_clean_dataframe_maps_y_to_binary() -> None:
    dataframe = pd.DataFrame({"y": ["yes", "no", "yes"]})

    cleaned = clean_dataframe(dataframe)

    assert cleaned["y"].tolist() == [1, 0, 1]


def test_validate_feature_columns_raises_for_missing_column() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [58],
            "balance": [2143],
        }
    )

    with pytest.raises(DataValidationError, match="Missing required feature columns"):
        validate_feature_columns(dataframe)
