"""
Model training — Isolation Forest, One-Class SVM, and feature scaler persistence.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.config import (
    ISO_FOREST_PARAMS,
    ISO_MODEL_NAME,
    MODELS_DIR,
    OCSVM_PARAMS,
    SCALER_NAME,
    SVM_MODEL_NAME,
)

logger = logging.getLogger(__name__)


def fit_scaler(features: np.ndarray) -> StandardScaler:
    """
    Fit StandardScaler on feature matrix.

    Args:
        features: 2D array of shape (n_samples, n_features).

    Returns:
        Fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def train_isolation_forest(
    scaled_features: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> IsolationForest:
    """
    Train Isolation Forest on scaled features.

    Args:
        scaled_features: Scaled feature matrix.
        params: Hyperparameters; merges with ISO_FOREST_PARAMS defaults.

    Returns:
        Fitted IsolationForest model.
    """
    cfg = {**ISO_FOREST_PARAMS, **(params or {})}
    model = IsolationForest(**cfg)
    model.fit(scaled_features)
    logger.info("Isolation Forest trained (contamination=%s)", cfg.get("contamination"))
    return model


def train_one_class_svm(
    scaled_features: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> OneClassSVM:
    """
    Train One-Class SVM on scaled features.

    Args:
        scaled_features: Scaled feature matrix.
        params: Hyperparameters; merges with OCSVM_PARAMS defaults.

    Returns:
        Fitted OneClassSVM model.
    """
    cfg = {**OCSVM_PARAMS, **(params or {})}
    model = OneClassSVM(**cfg)
    model.fit(scaled_features)
    logger.info("One-Class SVM trained (nu=%s)", cfg.get("nu"))
    return model


def train_models(
    features: np.ndarray,
) -> Tuple[StandardScaler, IsolationForest, OneClassSVM, np.ndarray]:
    """
    Fit scaler and both anomaly detection models.

    Args:
        features: Unscaled feature matrix (n_samples, n_features).

    Returns:
        Tuple of (scaler, isolation_forest, one_class_svm, scaled_features).
    """
    scaler = fit_scaler(features)
    scaled = scaler.transform(features)
    iso_model = train_isolation_forest(scaled)
    svm_model = train_one_class_svm(scaled)
    return scaler, iso_model, svm_model, scaled


def save_models(
    scaler: StandardScaler,
    iso_model: IsolationForest,
    svm_model: OneClassSVM,
    models_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Persist scaler and models to models/ using joblib.

    Returns:
        Dict mapping artifact name to file path.
    """
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "scaler": models_dir / SCALER_NAME,
        "isolation_forest": models_dir / ISO_MODEL_NAME,
        "one_class_svm": models_dir / SVM_MODEL_NAME,
    }

    joblib.dump(scaler, paths["scaler"])
    joblib.dump(iso_model, paths["isolation_forest"])
    joblib.dump(svm_model, paths["one_class_svm"])

    logger.info("Models saved to %s", models_dir)
    return paths


def load_models(
    models_dir: Optional[Path] = None,
) -> Tuple[StandardScaler, IsolationForest, OneClassSVM]:
    """
    Load scaler and trained models from models/.

    Returns:
        Tuple of (scaler, isolation_forest, one_class_svm).
    """
    models_dir = models_dir or MODELS_DIR
    scaler = joblib.load(models_dir / SCALER_NAME)
    iso_model = joblib.load(models_dir / ISO_MODEL_NAME)
    svm_model = joblib.load(models_dir / SVM_MODEL_NAME)
    logger.info("Models loaded from %s", models_dir)
    return scaler, iso_model, svm_model
