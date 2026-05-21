"""
Data preprocessing — cleaning, temporal splits, and persistence.

Temporal splits prevent look-ahead bias: scaler and models must only see
historical (train) data; anomaly flags are produced on the held-out test window.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.config import PROCESSED_CSV_NAME, PROCESSED_DATA_DIR, SPLIT_COLUMN, TRAIN_RATIO

logger = logging.getLogger(__name__)


def clean_data(data: pd.DataFrame, reset_index: bool = True) -> pd.DataFrame:
    """
    Remove missing values and normalize the datetime index.

    Args:
        data: Raw OHLCV DataFrame (Date as index or column).
        reset_index: If True, move Date to a column for downstream processing.

    Returns:
        Cleaned DataFrame sorted by Date ascending.
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
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info("Cleaned data shape: %s", df.shape)
    return df


def temporal_train_test_split(
    data: pd.DataFrame,
    date_col: str = "Date",
    train_ratio: float = TRAIN_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Chronological train/test split (no shuffling).

    Args:
        data: DataFrame sorted by date with engineered features.
        date_col: Name of the date column.
        train_ratio: Fraction of rows for training (0 < ratio < 1).

    Returns:
        (train_df, test_df, split_info) where split_info contains indices and dates.

    Raises:
        ValueError: If not enough rows or invalid ratio.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    df = data.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    if n < 20:
        raise ValueError(f"Need at least 20 rows for temporal split, got {n}")

    split_idx = int(n * train_ratio)
    split_idx = max(1, min(split_idx, n - 1))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    split_info = {
        "split_idx": split_idx,
        "n_total": n,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_ratio": train_ratio,
        "train_start": str(train_df[date_col].iloc[0]),
        "train_end": str(train_df[date_col].iloc[-1]),
        "test_start": str(test_df[date_col].iloc[0]),
        "test_end": str(test_df[date_col].iloc[-1]),
    }

    logger.info(
        "Temporal split: train=%d rows (%s to %s), test=%d rows (%s to %s)",
        split_info["n_train"],
        split_info["train_start"],
        split_info["train_end"],
        split_info["n_test"],
        split_info["test_start"],
        split_info["test_end"],
    )
    return train_df, test_df, split_info


def label_split_column(data: pd.DataFrame, split_idx: int) -> pd.DataFrame:
    """
    Add Split column marking train vs test rows (for visualization and exports).
    """
    df = data.copy()
    df[SPLIT_COLUMN] = "test"
    df.loc[: split_idx - 1, SPLIT_COLUMN] = "train"
    return df


def save_processed_data(
    data: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    """Save processed data to data/processed/."""
    path = path or (PROCESSED_DATA_DIR / PROCESSED_CSV_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    logger.info("Processed data saved to %s", path)
    return path


def load_processed_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load processed CSV from data/processed/."""
    path = path or (PROCESSED_DATA_DIR / PROCESSED_CSV_NAME)
    df = pd.read_csv(path, parse_dates=["Date"])
    logger.info("Loaded processed data from %s (%d rows)", path, len(df))
    return df
