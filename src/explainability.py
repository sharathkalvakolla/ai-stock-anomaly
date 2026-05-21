"""
Anomaly explainability — decision scores and z-score feature contributions.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.anomaly_detection import get_anomaly_scores
from src.config import EXPLAIN_TOP_K, FEATURE_COLUMNS, SPLIT_COLUMN

logger = logging.getLogger(__name__)


def compute_z_scores(
    scaled_row: np.ndarray,
    train_scaled: np.ndarray,
) -> np.ndarray:
    """
    Z-score of a scaled feature vector vs training distribution.

    Uses train-set mean/std of scaled features (post-StandardScaler, ~0 mean on train).
    """
    train_mean = np.mean(train_scaled, axis=0)
    train_std = np.std(train_scaled, axis=0)
    train_std = np.where(train_std < 1e-8, 1e-8, train_std)
    return (scaled_row - train_mean) / train_std


def top_contributing_features(
    z_scores: np.ndarray,
    feature_names: List[str],
    top_k: int = EXPLAIN_TOP_K,
) -> List[tuple]:
    """
    Rank features by absolute z-score (largest deviation from train norm).

    Returns:
        List of (feature_name, z_score) tuples, descending by |z|.
    """
    ranked_idx = np.argsort(np.abs(z_scores))[::-1][:top_k]
    return [(feature_names[i], float(z_scores[i])) for i in ranked_idx]


def explain_single_row(
    scaled_row: np.ndarray,
    train_scaled: np.ndarray,
    feature_names: List[str],
    iso_score: float,
    svm_score: float,
    top_k: int = EXPLAIN_TOP_K,
) -> dict:
    """
    Build explanation dict for one anomaly point.
    """
    z = compute_z_scores(scaled_row, train_scaled)
    contributors = top_contributing_features(z, feature_names, top_k)

    return {
        "iso_score": float(iso_score),
        "svm_score": float(svm_score),
        "top_features": contributors,
        "explanation_text": _format_explanation(contributors, iso_score, svm_score),
    }


def _format_explanation(
    contributors: List[tuple],
    iso_score: float,
    svm_score: float,
) -> str:
    """Human-readable one-line explanation."""
    parts = [f"{name} (z={z:+.2f})" for name, z in contributors]
    feat_str = ", ".join(parts) if parts else "n/a"
    return (
        f"ISO score={iso_score:.4f}, SVM score={svm_score:.4f}. "
        f"Top drivers: {feat_str}"
    )


def build_explanations_dataframe(
    results: pd.DataFrame,
    scaled_features_all: np.ndarray,
    train_scaled: np.ndarray,
    feature_names: Optional[List[str]] = None,
    anomaly_col: str = "Iso_Anomaly",
    model: str = "iso",
) -> pd.DataFrame:
    """
    Explanations for all anomaly rows on the test set.

    Args:
        results: Full results with Split, anomaly flags, scores.
        scaled_features_all: Scaled features aligned with results rows.
        train_scaled: Scaled training matrix for z-score reference.
        feature_names: Feature column names.
        anomaly_col: Which anomaly flag column to use.
        model: 'iso' or 'svm' for score column selection.

    Returns:
        DataFrame with Date, scores, top features, and explanation text.
    """
    feature_names = feature_names or FEATURE_COLUMNS
    test_mask = results[SPLIT_COLUMN] == "test" if SPLIT_COLUMN in results.columns else pd.Series(True, index=results.index)
    anomaly_mask = (results[anomaly_col] == 1) & test_mask

    if not anomaly_mask.any():
        return pd.DataFrame(columns=["Date", "Iso_Score", "SVM_Score", "Top_Features", "Explanation"])

    rows = []
    iso_scores, svm_scores = (
        results["Iso_Score"].values,
        results["SVM_Score"].values,
    )

    for idx in results.index[anomaly_mask]:
        i = results.index.get_loc(idx)
        scaled_row = scaled_features_all[i]
        expl = explain_single_row(
            scaled_row,
            train_scaled,
            feature_names,
            iso_scores[i] if not np.isnan(iso_scores[i]) else 0.0,
            svm_scores[i] if not np.isnan(svm_scores[i]) else 0.0,
        )
        top_feat_str = "; ".join(f"{n}: z={z:+.2f}" for n, z in expl["top_features"])
        rows.append(
            {
                "Date": results.loc[idx, "Date"],
                "Close": results.loc[idx, "Close"],
                "Iso_Score": iso_scores[i],
                "SVM_Score": svm_scores[i],
                "Top_Features": top_feat_str,
                "Explanation": expl["explanation_text"],
            }
        )

    return pd.DataFrame(rows)


def get_scaled_matrices(
    features_train: np.ndarray,
    features_test: np.ndarray,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform train and test with fitted scaler."""
    return scaler.transform(features_train), scaler.transform(features_test)
