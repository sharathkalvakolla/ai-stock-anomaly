"""
Anomaly detection — run inference and export results.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.config import OUTPUTS_DIR, RESULTS_CSV_NAME

logger = logging.getLogger(__name__)


def _sklearn_labels_to_binary(predictions: np.ndarray) -> np.ndarray:
    """
    Map sklearn anomaly labels to binary flags (1 = anomaly, 0 = normal).

    sklearn uses 1 for inliers and -1 for outliers.
    """
    return np.where(predictions == -1, 1, 0)


def predict_anomalies(
    scaled_features: np.ndarray,
    iso_model: IsolationForest,
    svm_model: OneClassSVM,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run both models on scaled features.

    Args:
        scaled_features: Scaled feature matrix.
        iso_model: Fitted Isolation Forest.
        svm_model: Fitted One-Class SVM.

    Returns:
        Tuple of (iso_anomaly_flags, svm_anomaly_flags) as 0/1 arrays.
    """
    iso_flags = _sklearn_labels_to_binary(iso_model.predict(scaled_features))
    svm_flags = _sklearn_labels_to_binary(svm_model.predict(scaled_features))
    return iso_flags, svm_flags


def attach_predictions(
    data: pd.DataFrame,
    iso_flags: np.ndarray,
    svm_flags: np.ndarray,
) -> pd.DataFrame:
    """
    Add anomaly columns to the main DataFrame.

    Args:
        data: DataFrame aligned row-wise with predictions.
        iso_flags: Isolation Forest binary flags.
        svm_flags: One-Class SVM binary flags.

    Returns:
        DataFrame with Iso_Anomaly and SVM_Anomaly columns.
    """
    result = data.copy()
    result["Iso_Anomaly"] = iso_flags
    result["SVM_Anomaly"] = svm_flags
    return result


def save_results(
    data: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save anomaly detection results to outputs/.

    Args:
        data: DataFrame including anomaly flags.
        output_path: Custom CSV path.

    Returns:
        Path to saved file.
    """
    output_path = output_path or (OUTPUTS_DIR / RESULTS_CSV_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)
    return output_path


def summarize_anomalies(data: pd.DataFrame) -> dict:
    """
    Compute summary statistics for detected anomalies.

    Returns:
        Dict with total points and counts per model.
    """
    return {
        "total_points": len(data),
        "isolation_forest_anomalies": int(data["Iso_Anomaly"].sum()),
        "one_class_svm_anomalies": int(data["SVM_Anomaly"].sum()),
    }


def run_detection(
    data: pd.DataFrame,
    features: np.ndarray,
    scaler: StandardScaler,
    iso_model: IsolationForest,
    svm_model: OneClassSVM,
    save: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    End-to-end detection: scale, predict, attach, optionally save.

    Args:
        data: Full DataFrame with Date and OHLCV.
        features: Unscaled feature matrix.
        scaler: Fitted StandardScaler.
        iso_model: Fitted Isolation Forest.
        svm_model: Fitted One-Class SVM.
        save: Write CSV to outputs/ if True.

    Returns:
        Tuple of (results DataFrame, summary dict).
    """
    scaled = scaler.transform(features)
    iso_flags, svm_flags = predict_anomalies(scaled, iso_model, svm_model)
    results = attach_predictions(data, iso_flags, svm_flags)
    summary = summarize_anomalies(results)

    if save:
        save_results(results)

    logger.info(
        "Detection complete - ISO: %d anomalies, SVM: %d anomalies (of %d points)",
        summary["isolation_forest_anomalies"],
        summary["one_class_svm_anomalies"],
        summary["total_points"],
    )
    return results, summary
