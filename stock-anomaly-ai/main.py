"""
Stock Anomaly AI — main entry point for the anomaly detection pipeline.

Usage:
    python main.py
    python main.py --symbol MSFT --start 2019-01-01 --end 2023-12-31
    python main.py --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as script
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection import run_detection
from src.config import DEFAULT_END_DATE, DEFAULT_START_DATE, DEFAULT_SYMBOL
from src.data_collection import download_stock_data
from src.feature_engineering import engineer_features, extract_feature_matrix
from src.model_training import save_models, train_models
from src.preprocessing import clean_data, save_processed_data
from src.utils import ensure_project_dirs, setup_logging
from src.visualization import generate_all_plots


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Stock price anomaly detection using Isolation Forest and One-Class SVM.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_SYMBOL,
        help=f"Stock ticker symbol (default: {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START_DATE,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=DEFAULT_END_DATE,
        help=f"End date YYYY-MM-DD (default: {DEFAULT_END_DATE})",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib visualizations",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plot images to outputs/",
    )
    return parser.parse_args()


def run_pipeline(
    symbol: str = DEFAULT_SYMBOL,
    start: str = DEFAULT_START_DATE,
    end: str = DEFAULT_END_DATE,
    show_plots: bool = True,
    save_plots: bool = False,
) -> dict:
    """
    Execute the full ML pipeline: collect → preprocess → features → train → detect → visualize.

    Returns:
        Summary dict with anomaly counts.
    """
    logger = setup_logging()
    ensure_project_dirs()

    logger.info("Starting pipeline for %s (%s to %s)", symbol, start, end)

    # 1. Data collection
    raw_data = download_stock_data(symbol=symbol, start=start, end=end, save=True)

    # 2. Preprocessing
    cleaned = clean_data(raw_data, reset_index=True)
    save_processed_data(cleaned)

    # 3. Feature engineering
    featured = engineer_features(cleaned)
    features_df = extract_feature_matrix(featured)
    feature_array = features_df.values

    # 4. Model training
    scaler, iso_model, svm_model, _ = train_models(feature_array)
    save_models(scaler, iso_model, svm_model)

    # 5. Anomaly detection
    results, summary = run_detection(
        data=featured,
        features=feature_array,
        scaler=scaler,
        iso_model=iso_model,
        svm_model=svm_model,
        save=True,
    )

    # 6. Visualization
    if show_plots:
        generate_all_plots(results, save=save_plots, show=True)

    logger.info("Pipeline finished successfully.")
    logger.info("Summary: %s", summary)
    return summary


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    summary = run_pipeline(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        show_plots=not args.no_plots,
        save_plots=args.save_plots,
    )

    print("\n" + "=" * 50)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 50)
    print(f"Total data points:          {summary['total_points']}")
    print(f"Isolation Forest anomalies: {summary['isolation_forest_anomalies']}")
    print(f"One-Class SVM anomalies:    {summary['one_class_svm_anomalies']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
