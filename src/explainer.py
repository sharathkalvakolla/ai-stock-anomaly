from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def _normal_ranges(train_df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    ranges: Dict[str, Tuple[float, float, float, float]] = {}
    for feature in feature_columns:
        mean = float(train_df[feature].mean())
        std = float(train_df[feature].std(ddof=0))
        std = max(std, 1e-8)
        ranges[feature] = (mean - std, mean + std, mean, std)
    return ranges


def _signal_for_feature(feature: str, value: float, std_away: float) -> str:
    if feature == "RSI" and value > 70:
        return "overbought signal"
    if feature == "RSI" and value < 30:
        return "oversold signal"
    if feature == "Volume_MA_20" and value > 2:
        return "unusual trading activity"
    if feature == "BB_Position" and value > 1:
        return "price broke above upper band"
    if feature == "BB_Position" and value < 0:
        return "price broke below lower band"
    if feature == "MACD_Hist" and value > 0.5:
        return "strong bullish momentum"
    if feature == "MACD_Hist" and value < -0.5:
        return "strong bearish momentum"
    if feature == "Daily_Return" and value < -0.03:
        return "sharp price drop"
    if feature == "Daily_Return" and value > 0.03:
        return "sharp price spike"
    if feature == "Volatility" and std_away > 1.5:
        return "elevated volatility"
    return "statistical deviation"


def classify_anomaly_type(row: pd.Series, both_models_agree: bool) -> str:
    rsi = float(row.get("RSI", 50.0))
    ret = float(row.get("Daily_Return", 0.0))
    vol_spike = float(row.get("Volume_MA_20", 1.0)) > 2.0
    vol = float(row.get("Volatility", 0.0))
    if rsi > 70 and ret > 0:
        return "Overbought / momentum event"
    if rsi < 30 and ret < 0:
        return "Oversold / panic selling"
    if vol_spike and abs(ret) > 0.03:
        return "Earnings / news event"
    if vol > 0.03 and both_models_agree:
        return "Market stress event"
    return "Statistical outlier"


def build_ai_explanations(
    results: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_columns: List[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or FEATURE_COLUMNS
    normal_ranges = _normal_ranges(train_df, feature_columns)
    anomaly_mask = ((results["Iso_Anomaly"] == 1) | (results["SVM_Anomaly"] == 1)) & (results["Split"] == "test")
    anomaly_rows = results.loc[anomaly_mask].copy()
    if anomaly_rows.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for _, row in anomaly_rows.iterrows():
        per_feature: List[Dict[str, Any]] = []
        for feature in feature_columns:
            value = float(row.get(feature, np.nan))
            low, high, mean, std = normal_ranges[feature]
            deviation = (value - mean) / std if np.isfinite(value) else 0.0
            signal = _signal_for_feature(feature, value, abs(deviation))
            per_feature.append(
                {
                    "feature": feature,
                    "value": value,
                    "normal_range": f"{low:.3f} to {high:.3f}",
                    "deviation": deviation,
                    "signal": signal,
                }
            )

        per_feature_sorted = sorted(per_feature, key=lambda x: abs(float(x["deviation"])), reverse=True)
        top5 = per_feature_sorted[:5]
        both = bool(row.get("Iso_Anomaly", 0) == 1 and row.get("SVM_Anomaly", 0) == 1)
        anomaly_type = classify_anomaly_type(row, both_models_agree=both)
        explanation_lines = [
            f"On {pd.to_datetime(row['Date']).date()}, {row.get('Symbol', 'selected stock')} was flagged as anomalous because:"
        ]
        for item in top5[:3]:
            explanation_lines.append(
                f"- {item['feature']} was {item['value']:.3f} (normal: {item['normal_range']}) -> {item['signal']}"
            )
        explanation_lines.append(f"Anomaly type: {anomaly_type}")

        records.append(
            {
                "Date": row["Date"],
                "Iso_Score": float(row.get("Iso_Score", np.nan)),
                "SVM_Score": float(row.get("SVM_Score", np.nan)),
                "Anomaly_Type": anomaly_type,
                "Severity": "severe" if both else "moderate" if (row.get("Iso_Anomaly", 0) or row.get("SVM_Anomaly", 0)) else "mild",
                "Feature_Details": per_feature_sorted,
                "Top5": top5,
                "Plain_English_Explanation": "\n".join(explanation_lines),
            }
        )

    return pd.DataFrame(records).sort_values("Date")
