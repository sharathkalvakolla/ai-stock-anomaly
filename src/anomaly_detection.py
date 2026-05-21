"""
Anomaly detection — temporal test-set inference (no train-set leakage in flags).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.config import OUTPUTS_DIR, RESULTS_CSV_NAME, SPLIT_COLUMN

logger = logging.getLogger(__name__)


def _sklearn_labels_to_binary(predictions: np.ndarray) -> np.ndarray:
    """Map sklearn labels: -1 (outlier) -> 1, 1 (inlier) -> 0."""
    return np.where(predictions == -1, 1, 0)


def predict_anomalies(
    scaled_features: np.ndarray,
    iso_model: IsolationForest,
    svm_model: OneClassSVM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run both models on scaled features."""
    iso_flags = _sklearn_labels_to_binary(iso_model.predict(scaled_features))
    svm_flags = _sklearn_labels_to_binary(svm_model.predict(scaled_features))
    return iso_flags, svm_flags


def get_anomaly_scores(
    scaled_features: np.ndarray,
    iso_model: IsolationForest,
    svm_model: OneClassSVM,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decision scores for explainability.

    Isolation Forest: score_samples (lower = more anomalous).
    One-Class SVM: decision_function (negative = anomaly).
    """
    iso_scores = iso_model.score_samples(scaled_features)
    svm_scores = svm_model.decision_function(scaled_features)
    return iso_scores, svm_scores


def attach_predictions(
    data: pd.DataFrame,
    iso_flags: np.ndarray,
    svm_flags: np.ndarray,
    iso_scores: Optional[np.ndarray] = None,
    svm_scores: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Add anomaly columns and optional score columns."""
    result = data.copy()
    result["Iso_Anomaly"] = iso_flags
    result["SVM_Anomaly"] = svm_flags
    if iso_scores is not None:
        result["Iso_Score"] = iso_scores
    if svm_scores is not None:
        result["SVM_Score"] = svm_scores
    return result


def run_detection_temporal(
    full_data: pd.DataFrame,
    features_train: np.ndarray,
    features_test: np.ndarray,
    test_row_indices: np.ndarray,
    scaler: StandardScaler,
    iso_model: IsolationForest,
    svm_model: OneClassSVM,
    split_idx: int,
    save: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Detect anomalies on the test window only (prevents inflated in-sample flags).

    Train rows receive anomaly flag 0; test rows receive model predictions.
    Scaler transform uses statistics learned on train only.

    Args:
        full_data: Full featured DataFrame (train + test), sorted by Date.
        features_train: Train feature matrix (used only for logging consistency).
        features_test: Test feature matrix for prediction.
        test_row_indices: Integer positions in full_data for test rows.
        scaler: Fitted on train only.
        iso_model, svm_model: Fitted on train only.
        split_idx: Index where test window begins.
        save: Write CSV to outputs/.

    Returns:
        (results DataFrame, summary dict)
    """
    n = len(full_data)
    iso_flags = np.zeros(n, dtype=int)
    svm_flags = np.zeros(n, dtype=int)
    iso_scores_full = np.full(n, np.nan)
    svm_scores_full = np.full(n, np.nan)

    scaled_test = scaler.transform(features_test)
    iso_test, svm_test = predict_anomalies(scaled_test, iso_model, svm_model)
    iso_scores_test, svm_scores_test = get_anomaly_scores(scaled_test, iso_model, svm_model)

    iso_flags[test_row_indices] = iso_test
    svm_flags[test_row_indices] = svm_test
    iso_scores_full[test_row_indices] = iso_scores_test
    svm_scores_full[test_row_indices] = svm_scores_test

    results = attach_predictions(
        full_data,
        iso_flags,
        svm_flags,
        iso_scores=iso_scores_full,
        svm_scores=svm_scores_full,
    )
    results = results.copy()
    results[SPLIT_COLUMN] = "test"
    split_col_idx = results.columns.get_loc(SPLIT_COLUMN)
    # Use positional assignment to avoid index-label drift when upstream rows were dropped.
    results.iloc[:split_idx, split_col_idx] = "train"

    test_mask = results[SPLIT_COLUMN] == "test"
    summary = {
        "total_points": n,
        "train_points": int((~test_mask).sum()),
        "test_points": int(test_mask.sum()),
        "isolation_forest_anomalies": int(results.loc[test_mask, "Iso_Anomaly"].sum()),
        "one_class_svm_anomalies": int(results.loc[test_mask, "SVM_Anomaly"].sum()),
        "both_models_anomalies": int(
            ((results["Iso_Anomaly"] == 1) & (results["SVM_Anomaly"] == 1) & test_mask).sum()
        ),
        "iso_only_anomalies": int(
            ((results["Iso_Anomaly"] == 1) & (results["SVM_Anomaly"] == 0) & test_mask).sum()
        ),
        "svm_only_anomalies": int(
            ((results["Iso_Anomaly"] == 0) & (results["SVM_Anomaly"] == 1) & test_mask).sum()
        ),
    }

    if save:
        save_results(results)

    logger.info(
        "Temporal detection (test only) - ISO: %d, SVM: %d, both: %d (test n=%d)",
        summary["isolation_forest_anomalies"],
        summary["one_class_svm_anomalies"],
        summary["both_models_anomalies"],
        summary["test_points"],
    )
    return results, summary


def run_detection(
    data: pd.DataFrame,
    features: np.ndarray,
    scaler: StandardScaler,
    iso_model: IsolationForest,
    svm_model: OneClassSVM,
    save: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Legacy full-sample detection (may inflate metrics due to in-sample training).
    Prefer run_detection_temporal for honest evaluation.
    """
    scaled = scaler.transform(features)
    iso_flags, svm_flags = predict_anomalies(scaled, iso_model, svm_model)
    iso_scores, svm_scores = get_anomaly_scores(scaled, iso_model, svm_model)
    results = attach_predictions(data, iso_flags, svm_flags, iso_scores, svm_scores)
    summary = summarize_anomalies(results)
    if save:
        save_results(results)
    return results, summary


def save_results(data: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Save anomaly detection results to outputs/."""
    output_path = output_path or (OUTPUTS_DIR / RESULTS_CSV_NAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)
    return output_path


def summarize_anomalies(data: pd.DataFrame) -> dict:
    """Summary statistics; respects test-only flags when Split column exists."""
    if SPLIT_COLUMN in data.columns:
        test_df = data[data[SPLIT_COLUMN] == "test"]
    else:
        test_df = data

    both = int(((test_df["Iso_Anomaly"] == 1) & (test_df["SVM_Anomaly"] == 1)).sum())
    return {
        "total_points": len(data),
        "train_points": int((data[SPLIT_COLUMN] == "train").sum()) if SPLIT_COLUMN in data.columns else 0,
        "test_points": len(test_df),
        "isolation_forest_anomalies": int(test_df["Iso_Anomaly"].sum()),
        "one_class_svm_anomalies": int(test_df["SVM_Anomaly"].sum()),
        "both_models_anomalies": both,
        "iso_only_anomalies": int(
            ((test_df["Iso_Anomaly"] == 1) & (test_df["SVM_Anomaly"] == 0)).sum()
        ),
        "svm_only_anomalies": int(
            ((test_df["Iso_Anomaly"] == 0) & (test_df["SVM_Anomaly"] == 1)).sum()
        ),
    }
