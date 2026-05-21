"""
Feature engineering — technical indicators and context features for anomaly detection.
Includes: Daily Return, Moving Averages, Volatility, Volume Change,
          RSI, MACD, Bollinger Bands, Volume MA.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, ROLLING_WINDOWS

logger = logging.getLogger(__name__)


# ─── Original features ────────────────────────────────────────────────────────

def add_daily_return(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Percentage daily return from close price."""
    df = df.copy()
    df["Daily_Return"] = df[price_col].pct_change()
    return df


def add_moving_averages(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Short (10) and long (20) day moving averages."""
    df = df.copy()
    df["MA_10"] = df[price_col].rolling(window=ROLLING_WINDOWS["ma_short"]).mean()
    df["MA_20"] = df[price_col].rolling(window=ROLLING_WINDOWS["ma_long"]).mean()
    return df


def add_volatility_features(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Rolling volatility and rolling standard deviation."""
    df = df.copy()
    df["Volatility"] = df[price_col].rolling(window=ROLLING_WINDOWS["volatility"]).std()
    df["Rolling_STD"] = df[price_col].rolling(window=ROLLING_WINDOWS["rolling_std"]).std()
    return df


def add_volume_change(df: pd.DataFrame, volume_col: str = "Volume") -> pd.DataFrame:
    """Percentage change in trading volume."""
    df = df.copy()
    df["Volume_Change"] = df[volume_col].pct_change()
    return df


# ─── New technical indicators ─────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Relative Strength Index (RSI) — 14-day default.

    RSI measures momentum:
    - RSI > 70 → overbought (price may drop soon) → potential anomaly
    - RSI < 30 → oversold (price may rise soon) → potential anomaly
    - Range: 0 to 100
    """
    df = df.copy()
    period = ROLLING_WINDOWS["rsi"]
    delta = df[price_col].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)  # neutral fill for early NaNs
    return df


def add_macd(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).

    Components:
    - MACD Line     = EMA(12) - EMA(26)   → trend direction
    - Signal Line   = EMA(9) of MACD      → trigger line
    - Histogram     = MACD - Signal       → momentum strength

    Anomaly signals:
    - Large histogram spike → unusual momentum
    - MACD crossing signal  → trend reversal
    """
    df = df.copy()
    fast = ROLLING_WINDOWS["macd_fast"]    # 12
    slow = ROLLING_WINDOWS["macd_slow"]    # 26
    signal = ROLLING_WINDOWS["macd_signal"]  # 9

    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Bollinger Bands — 20-day SMA ± 2 standard deviations.

    Components:
    - BB_Upper    = SMA(20) + 2*STD(20)   → upper boundary
    - BB_Lower    = SMA(20) - 2*STD(20)   → lower boundary
    - BB_Width    = (Upper - Lower) / SMA → band width (volatility measure)
    - BB_Position = (Close - Lower) / (Upper - Lower) → where price sits (0=bottom, 1=top)

    Anomaly signals:
    - Price outside bands (BB_Position < 0 or > 1) → strong anomaly
    - BB_Width spike → volatility explosion
    """
    df = df.copy()
    period = ROLLING_WINDOWS["bb_period"]  # 20

    sma = df[price_col].rolling(window=period).mean()
    std = df[price_col].rolling(window=period).std()

    df["BB_Upper"] = sma + (2 * std)
    df["BB_Lower"] = sma - (2 * std)

    band_range = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    df["BB_Width"] = band_range / sma
    df["BB_Position"] = (df[price_col] - df["BB_Lower"]) / band_range

    # Fill edge NaNs with neutral values
    df["BB_Width"] = df["BB_Width"].fillna(0)
    df["BB_Position"] = df["BB_Position"].fillna(0.5)
    return df


def add_volume_ma(df: pd.DataFrame, volume_col: str = "Volume") -> pd.DataFrame:
    """
    20-day Volume Moving Average.
    Compares current volume to recent average — spikes signal unusual activity.
    """
    df = df.copy()
    period = ROLLING_WINDOWS["volume_ma"]  # 20
    vol_ma = df[volume_col].rolling(window=period).mean()
    # Ratio: current volume vs average (>2 means double the usual volume)
    df["Volume_MA_20"] = df[volume_col] / vol_ma.replace(0, np.nan)
    df["Volume_MA_20"] = df["Volume_MA_20"].fillna(1)
    return df


# ─── Main pipeline ────────────────────────────────────────────────────────────

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Applies all original + new technical indicator features.

    Args:
        data: Cleaned OHLCV DataFrame with Date column.

    Returns:
        DataFrame with all engineered features, NaN rows removed.
    """
    df = data.copy()

    # Original features
    df = add_daily_return(df)
    df = add_moving_averages(df)
    df = add_volatility_features(df)
    df = add_volume_change(df)

    # New technical indicators
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volume_ma(df)

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
        feature_columns: Defaults to config FEATURE_COLUMNS (all 15 features).

    Returns:
        Feature-only DataFrame.
    """
    feature_columns = feature_columns or FEATURE_COLUMNS
    missing = set(feature_columns) - set(data.columns)
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return data[feature_columns].copy()