"""
End-to-end pipeline orchestration with temporal integrity.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.anomaly_detection import run_detection_temporal
from src.config import FEATURE_COLUMNS, TRAIN_RATIO
from src.data_collection import download_stock_data
from src.feature_engineering import engineer_features, extract_feature_matrix
from src.model_training import (
    artifacts_exist,
    build_training_metadata,
    load_models,
    load_training_metadata,
    models_match_request,
    remove_stale_model_artifacts,
    save_models,
    train_models_on_train_split,
)
from src.preprocessing import clean_data, save_processed_data, temporal_train_test_split

logger = logging.getLogger(__name__)


def run_temporal_pipeline(
    symbol: str,
    start: str,
    end: str,
    train_ratio: float = TRAIN_RATIO,
    force_retrain: bool = False,
    save: bool = True,
) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Full pipeline with chronological train/test split.

    Returns:
        (results_df, summary_dict, split_info)
    """
    remove_stale_model_artifacts()
    raw = download_stock_data(symbol=symbol, start=start, end=end, save=save)
    cleaned = clean_data(raw, reset_index=True)
    if save:
        save_processed_data(cleaned)

    featured = engineer_features(cleaned)
    train_df, test_df, split_info = temporal_train_test_split(featured, train_ratio=train_ratio)

    features_train = extract_feature_matrix(train_df).values
    features_test = extract_feature_matrix(test_df).values
    n_train = len(train_df)
    test_row_indices = np.arange(n_train, len(featured))

    metadata = build_training_metadata(symbol, start, end, split_info, FEATURE_COLUMNS, train_ratio)

    if not force_retrain and artifacts_exist():
        existing = load_training_metadata()
        if models_match_request(existing, symbol, start, end, train_ratio):
            scaler, iso_model, svm_model = load_models()
            logger.info("Loaded cached models for %s", symbol)
        else:
            force_retrain = True

    if force_retrain or not artifacts_exist():
        scaler, iso_model, svm_model = train_models_on_train_split(features_train)
        if save:
            save_models(scaler, iso_model, svm_model, metadata=metadata)

    results, summary = run_detection_temporal(
        full_data=featured,
        features_train=features_train,
        features_test=features_test,
        test_row_indices=test_row_indices,
        scaler=scaler,
        iso_model=iso_model,
        svm_model=svm_model,
        split_idx=n_train,
        save=save,
    )

    return results, summary, split_info
