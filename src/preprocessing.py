"""
Data preprocessing — cleaning, validation, and persistence of processed data.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import PROCESSED_CSV_NAME, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def clean_data(data: pd.DataFrame, reset_index: bool = True) -> pd.DataFrame:
    """
    Remove missing values and normalize the datetime index.

    Args:
        data: Raw OHLCV DataFrame (Date as index or column).
        reset_index: If True, move Date to a column for downstream processing.

    Returns:
        Cleaned DataFrame.
    """
    df = data.copy()

    if "Date" not in df.columns:
        df.index = pd.to_datetime(df.index)
        if reset_index:
            df = df.reset_index()
            if "index" in df.columns and "Date" not in df.columns:
                df = df.rename(columns={"index": "Date"})
    else:
        df["Date"] = pd.to_datetime(df["Date"])

    df = df.dropna()

    logger.info("Cleaned data shape: %s", df.shape)
    return df


def save_processed_data(
    data: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    """
    Save processed (cleaned) data to data/processed/.

    Args:
        data: Cleaned DataFrame.
        path: Output CSV path; defaults to data/processed/stock_processed.csv.

    Returns:
        Path where the file was written.
    """
    path = path or (PROCESSED_DATA_DIR / PROCESSED_CSV_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    logger.info("Processed data saved to %s", path)
    return path


def load_processed_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load processed CSV from data/processed/.

    Args:
        path: Path to processed CSV.

    Returns:
        DataFrame with parsed Date column.
    """
    path = path or (PROCESSED_DATA_DIR / PROCESSED_CSV_NAME)
    df = pd.read_csv(path, parse_dates=["Date"])
    logger.info("Loaded processed data from %s (%d rows)", path, len(df))
    return df
