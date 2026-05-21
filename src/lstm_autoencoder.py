from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.config import LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_SEQUENCE_LENGTH, MODELS_DIR

logger = logging.getLogger(__name__)

LSTM_MODEL_DIR = MODELS_DIR / "lstm_autoencoder"


def _create_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    return np.array([data[i : i + seq_len] for i in range(len(data) - seq_len + 1)])


def _build_model(input_dim: int, seq_len: int):
    import tensorflow as tf
    from tensorflow.keras import Model, layers

    inputs = tf.keras.Input(shape=(seq_len, input_dim))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.RepeatVector(seq_len)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(input_dim))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_or_load_lstm(
    train_scaled: np.ndarray,
    force_retrain: bool = False,
    model_dir: Optional[Path] = None,
) -> Tuple[object, float]:
    model_dir = model_dir or LSTM_MODEL_DIR
    model_path = model_dir / "model.keras"
    threshold_path = model_dir / "threshold.npy"
    import tensorflow as tf

    if model_path.exists() and threshold_path.exists() and not force_retrain:
        model = tf.keras.models.load_model(model_path)
        threshold = float(np.load(threshold_path))
        return model, threshold

    model_dir.mkdir(parents=True, exist_ok=True)
    seq_train = _create_sequences(train_scaled, LSTM_SEQUENCE_LENGTH)
    model = _build_model(train_scaled.shape[1], LSTM_SEQUENCE_LENGTH)
    model.fit(
        seq_train,
        seq_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        verbose=0,
    )
    train_recon = model.predict(seq_train, verbose=0)
    train_err = np.mean(np.square(seq_train - train_recon), axis=(1, 2))
    threshold = float(np.mean(train_err) + (2 * np.std(train_err)))
    model.save(model_path)
    np.save(threshold_path, np.array(threshold))
    return model, threshold


def detect_lstm_anomalies(
    train_scaled: np.ndarray,
    full_scaled: np.ndarray,
    split_idx: int,
    force_retrain: bool = False,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    try:
        model, threshold = train_or_load_lstm(train_scaled, force_retrain=force_retrain)
        seq_all = _create_sequences(full_scaled, LSTM_SEQUENCE_LENGTH)
        recon = model.predict(seq_all, verbose=0)
        errs = np.mean(np.square(seq_all - recon), axis=(1, 2))

        scores_full = np.full(len(full_scaled), np.nan)
        flags_full = np.zeros(len(full_scaled), dtype=int)
        start_pos = LSTM_SEQUENCE_LENGTH - 1
        scores_full[start_pos:] = errs
        flags_full[(scores_full > threshold) & np.isfinite(scores_full)] = 1
        flags_full[:split_idx] = 0
        return flags_full, scores_full, None
    except Exception as exc:
        logger.warning("LSTM unavailable or failed: %s", exc)
        return np.zeros(len(full_scaled), dtype=int), np.full(len(full_scaled), np.nan), str(exc)
