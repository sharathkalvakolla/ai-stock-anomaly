"""
Feature engineering — technical indicators and context features for anomaly detection.
"""

import logging
from typing import List

import pandas as pd

from src.config import FEATURE_COLUMNS, ROLLING_WINDOWS

logger = logging.getLogger(__name__)


def add_daily_return(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Compute percentage daily return from close price."""
    df = df.copy()
    df["Daily_Return"] = df[price_col].pct_change()
    return df


def add_moving_averages(
    df: pd.DataFrame,
    price_col: str = "Close",
    short_window: int = None,
    long_window: int = None,
) -> pd.DataFrame:
    """Add short and long moving averages."""
    df = df.copy()
    short_window = short_window or ROLLING_WINDOWS["ma_short"]
    long_window = long_window or ROLLING_WINDOWS["ma_long"]
    df["MA_10"] = df[price_col].rolling(window=short_window).mean()
    df["MA_20"] = df[price_col].rolling(window=long_window).mean()
    return df


def add_volatility_features(
    df: pd.DataFrame,
    price_col: str = "Close",
    vol_window: int = None,
    std_window: int = None,
) -> pd.DataFrame:
    """Add rolling volatility and rolling standard deviation."""
    df = df.copy()
    vol_window = vol_window or ROLLING_WINDOWS["volatility"]
    std_window = std_window or ROLLING_WINDOWS["rolling_std"]
    df["Volatility"] = df[price_col].rolling(window=vol_window).std()
    df["Rolling_STD"] = df[price_col].rolling(window=std_window).std()
    return df


def add_volume_change(df: pd.DataFrame, volume_col: str = "Volume") -> pd.DataFrame:
    """Compute percentage change in trading volume."""
    df = df.copy()
    df["Volume_Change"] = df[volume_col].pct_change()
    return df


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline.

    Args:
        data: Cleaned OHLCV DataFrame with Date column.

    Returns:
        DataFrame with engineered features; rows with NaN from rolling windows removed.
    """
    df = data.copy()
    df = add_daily_return(df)
    df = add_moving_averages(df)
    df = add_volatility_features(df)
    df = add_volume_change(df)
    df = df.dropna()

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df


def extract_feature_matrix(
    data: pd.DataFrame,
    feature_columns: List[str] = None,
) -> pd.DataFrame:
    """
    Select feature columns for model input.

    Args:
        data: DataFrame containing engineered features.
        feature_columns: List of column names; defaults to config FEATURE_COLUMNS.

    Returns:
        Feature-only DataFrame.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    missing = set(feature_columns) - set(data.columns)
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return data[feature_columns].copy()
