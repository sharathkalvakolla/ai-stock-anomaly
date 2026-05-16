"""
Project-wide configuration: paths, model hyperparameters, and feature definitions.
"""

from pathlib import Path

# Project root (stock-anomaly-ai/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory layout
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Default data collection settings
DEFAULT_SYMBOL = "AAPL"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-01-01"
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Feature engineering
FEATURE_COLUMNS = [
    "Daily_Return",
    "MA_10",
    "MA_20",
    "Volatility",
    "Rolling_STD",
    "Volume_Change",
]

ROLLING_WINDOWS = {
    "ma_short": 10,
    "ma_long": 20,
    "volatility": 10,
    "rolling_std": 5,
}

# Isolation Forest
ISO_FOREST_PARAMS = {
    "n_estimators": 100,
    "contamination": 0.02,
    "random_state": 42,
}

# One-Class SVM
OCSVM_PARAMS = {
    "kernel": "rbf",
    "gamma": 0.001,
    "nu": 0.05,
}

# Output filenames
RAW_CSV_NAME = "stock_raw.csv"
PROCESSED_CSV_NAME = "stock_processed.csv"
RESULTS_CSV_NAME = "stock_anomaly_results.csv"
ISO_MODEL_NAME = "isolation_forest.joblib"
SVM_MODEL_NAME = "one_class_svm.joblib"
SCALER_NAME = "feature_scaler.joblib"
