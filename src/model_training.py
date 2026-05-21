"""
Model training — temporal fit (train window only), persistence, and metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.config import (
    FEATURE_COLUMNS,
    ISO_FOREST_PARAMS,
    ISO_MODEL_NAME,
    METADATA_NAME,
    MODELS_DIR,
    OCSVM_PARAMS,
    SCALER_NAME,
    SVM_MODEL_NAME,
    TRAIN_RATIO,
)

logger = logging.getLogger(__name__)


def remove_stale_model_artifacts(
    expected_feature_count: int = len(FEATURE_COLUMNS),
    models_dir: Optional[Path] = None,
) -> bool:
    """
    Remove incompatible saved artifacts (e.g., old 6-feature models).

    Returns:
        True when one or more files were deleted; False otherwise.
    """
    models_dir = models_dir or MODELS_DIR
    required_artifacts = [SCALER_NAME, ISO_MODEL_NAME, SVM_MODEL_NAME]
    artifact_paths = [models_dir / name for name in required_artifacts]

    if not all(path.exists() for path in artifact_paths):
        return False

    try:
        scaler = joblib.load(models_dir / SCALER_NAME)
        scaler_feature_count = int(getattr(scaler, "n_features_in_", 0))
    except Exception as exc:
        logger.warning("Failed to inspect saved scaler. Removing model artifacts. Error: %s", exc)
        scaler_feature_count = -1

    metadata = load_training_metadata(models_dir=models_dir)
    metadata_feature_count = (
        len(metadata.get("feature_columns", []))
        if isinstance(metadata, dict) and isinstance(metadata.get("feature_columns"), list)
        else 0
    )

    should_remove = (
        scaler_feature_count != expected_feature_count
        or (metadata_feature_count not in (0, expected_feature_count))
    )
    if not should_remove:
        return False

    removed_any = False
    for path in artifact_paths + [models_dir / METADATA_NAME]:
        if path.exists():
            try:
                path.unlink()
                removed_any = True
            except Exception as exc:
                logger.warning("Could not delete stale artifact %s: %s", path, exc)

    if removed_any:
        logger.info(
            "Removed stale model artifacts (expected %d features, scaler=%d, metadata=%d).",
            expected_feature_count,
            scaler_feature_count,
            metadata_feature_count,
        )
    return removed_any


def fit_scaler(features_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on training features only."""
    scaler = StandardScaler()
    scaler.fit(features_train)
    return scaler


def train_isolation_forest(
    scaled_features_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> IsolationForest:
    """Train Isolation Forest on scaled training data."""
    cfg = {**ISO_FOREST_PARAMS, **(params or {})}
    model = IsolationForest(**cfg)
    model.fit(scaled_features_train)
    logger.info("Isolation Forest trained on %d samples", len(scaled_features_train))
    return model


def train_one_class_svm(
    scaled_features_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> OneClassSVM:
    """Train One-Class SVM on scaled training data."""
    cfg = {**OCSVM_PARAMS, **(params or {})}
    model = OneClassSVM(**cfg)
    model.fit(scaled_features_train)
    logger.info("One-Class SVM trained on %d samples", len(scaled_features_train))
    return model


def train_models_on_train_split(
    features_train: np.ndarray,
) -> Tuple[StandardScaler, IsolationForest, OneClassSVM]:
    """
    Fit scaler and both models using only the temporal training window.

    Args:
        features_train: Unscaled feature matrix for train rows only.

    Returns:
        (scaler, isolation_forest, one_class_svm)
    """
    if len(features_train) < 10:
        raise ValueError("Training set too small; need at least 10 rows.")

    scaler = fit_scaler(features_train)
    scaled_train = scaler.transform(features_train)
    iso_model = train_isolation_forest(scaled_train)
    svm_model = train_one_class_svm(scaled_train)
    return scaler, iso_model, svm_model


def train_models(features: np.ndarray) -> Tuple[StandardScaler, IsolationForest, OneClassSVM, np.ndarray]:
    """
    Legacy: fit on all data. Prefer train_models_on_train_split for production.

    Returns:
        (scaler, iso, svm, scaled_all)
    """
    scaler = fit_scaler(features)
    scaled = scaler.transform(features)
    iso_model = train_isolation_forest(scaled)
    svm_model = train_one_class_svm(scaled)
    return scaler, iso_model, svm_model, scaled


def build_training_metadata(
    symbol: str,
    start: str,
    end: str,
    split_info: dict,
    feature_columns: list,
    train_ratio: float = TRAIN_RATIO,
) -> dict:
    """Metadata stored alongside joblib artifacts for cache validation."""
    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "train_ratio": train_ratio,
        "feature_columns": feature_columns,
        **split_info,
    }


def save_models(
    scaler: StandardScaler,
    iso_model: IsolationForest,
    svm_model: OneClassSVM,
    metadata: Optional[dict] = None,
    models_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Persist scaler, models, and optional metadata JSON."""
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "scaler": models_dir / SCALER_NAME,
        "isolation_forest": models_dir / ISO_MODEL_NAME,
        "one_class_svm": models_dir / SVM_MODEL_NAME,
        "metadata": models_dir / METADATA_NAME,
    }

    joblib.dump(scaler, paths["scaler"])
    joblib.dump(iso_model, paths["isolation_forest"])
    joblib.dump(svm_model, paths["one_class_svm"])

    if metadata is not None:
        paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Models saved to %s", models_dir)
    return paths


def load_models(
    models_dir: Optional[Path] = None,
) -> Tuple[StandardScaler, IsolationForest, OneClassSVM]:
    """Load scaler and trained models from models/."""
    models_dir = models_dir or MODELS_DIR
    scaler = joblib.load(models_dir / SCALER_NAME)
    iso_model = joblib.load(models_dir / ISO_MODEL_NAME)
    svm_model = joblib.load(models_dir / SVM_MODEL_NAME)
    logger.info("Models loaded from %s", models_dir)
    return scaler, iso_model, svm_model


def load_training_metadata(models_dir: Optional[Path] = None) -> Optional[dict]:
    """Load training metadata if present."""
    models_dir = models_dir or MODELS_DIR
    path = models_dir / METADATA_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def models_match_request(
    metadata: Optional[dict],
    symbol: str,
    start: str,
    end: str,
    train_ratio: float = TRAIN_RATIO,
) -> bool:
    """Return True if saved artifacts match the current analysis request."""
    if metadata is None:
        return False
    return (
        metadata.get("symbol") == symbol
        and metadata.get("start") == start
        and metadata.get("end") == end
        and float(metadata.get("train_ratio", -1)) == float(train_ratio)
    )


def artifacts_exist(models_dir: Optional[Path] = None) -> bool:
    """Check whether all required joblib files exist."""
    models_dir = models_dir or MODELS_DIR
    required = [SCALER_NAME, ISO_MODEL_NAME, SVM_MODEL_NAME]
    return all((models_dir / name).exists() for name in required)


def remove_stale_model_artifacts(models_dir: Optional[Path] = None) -> None:
    """Delete saved models if feature count doesn't match current config."""
    models_dir = models_dir or MODELS_DIR
    meta = load_training_metadata(models_dir)
    if meta is None:
        return
    saved_features = meta.get("feature_columns", [])
    if len(saved_features) != len(FEATURE_COLUMNS):
        import logging
        logging.getLogger(__name__).info(
            "Stale models detected (%d vs %d features) — deleting.",
            len(saved_features), len(FEATURE_COLUMNS),
        )
        for name in [SCALER_NAME, ISO_MODEL_NAME, SVM_MODEL_NAME, METADATA_NAME]:
            p = models_dir / name
            if p.exists():
                p.unlink()