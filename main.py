"""
Stock Anomaly AI — CLI entry point (temporal train/test pipeline).

Usage:
    python main.py
    python main.py --symbol MSFT --train-ratio 0.8 --force-retrain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_END_DATE, DEFAULT_START_DATE, DEFAULT_SYMBOL, TRAIN_RATIO
from src.pipeline import run_temporal_pipeline
from src.utils import ensure_project_dirs, setup_logging
from src.visualization import generate_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stock anomaly detection with temporal train/test split.",
    )
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--start", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end", type=str, default=DEFAULT_END_DATE)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Fraction of data for training (default: 0.75)",
    )
    parser.add_argument("--force-retrain", action="store_true", help="Ignore cached models")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    ensure_project_dirs()

    logger.info(
        "Temporal pipeline: %s (%s to %s), train_ratio=%.2f",
        args.symbol,
        args.start,
        args.end,
        args.train_ratio,
    )

    results, summary, split_info = run_temporal_pipeline(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        train_ratio=args.train_ratio,
        force_retrain=args.force_retrain,
    )

    if not args.no_plots:
        generate_all_plots(results, save=args.save_plots, show=True)

    print("\n" + "=" * 55)
    print("TEMPORAL ANOMALY DETECTION SUMMARY (test window)")
    print("=" * 55)
    print(f"Train rows:                 {summary.get('train_points', split_info['n_train'])}")
    print(f"Test rows:                  {summary.get('test_points', split_info['n_test'])}")
    print(f"Test period:                {split_info['test_start']} to {split_info['test_end']}")
    print(f"Isolation Forest anomalies: {summary['isolation_forest_anomalies']}")
    print(f"One-Class SVM anomalies:    {summary['one_class_svm_anomalies']}")
    print(f"Both models agree:          {summary.get('both_models_anomalies', 0)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
