"""
Data collection module — download OHLCV stock data via yFinance.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    OHLCV_COLUMNS,
    RAW_CSV_NAME,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


def download_stock_data(
    symbol: str = DEFAULT_SYMBOL,
    start: str = DEFAULT_START_DATE,
    end: str = DEFAULT_END_DATE,
    save: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL').
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        save: If True, persist raw CSV under data/raw/.
        output_path: Custom save path; defaults to data/raw/stock_raw.csv.

    Returns:
        DataFrame with Open, High, Low, Close, Volume columns.

    Raises:
        ValueError: If download returns empty data.
    """
    logger.info("Downloading %s from %s to %s", symbol, start, end)

    raw = yf.download(symbol, start=start, end=end, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'.")

    # Flatten MultiIndex columns when yfinance returns them for single tickers
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    data = raw[OHLCV_COLUMNS].copy()
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"

    logger.info("Downloaded %d rows for %s", len(data), symbol)

    if save:
        path = output_path or (RAW_DATA_DIR / RAW_CSV_NAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path)
        logger.info("Raw data saved to %s", path)

    return data


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load previously saved raw CSV from data/raw/.

    Args:
        path: Path to CSV file; defaults to configured raw file.

    Returns:
        DataFrame with Date as index.
    """
    path = path or (RAW_DATA_DIR / RAW_CSV_NAME)
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    logger.info("Loaded raw data from %s (%d rows)", path, len(df))
    return df
