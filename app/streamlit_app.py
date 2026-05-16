"""
Stock Anomaly AI — Streamlit dashboard.

Run from project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly_detection import run_detection, summarize_anomalies
from src.config import DEFAULT_END_DATE, DEFAULT_START_DATE, DEFAULT_SYMBOL
from src.data_collection import download_stock_data
from src.feature_engineering import engineer_features, extract_feature_matrix
from src.model_training import save_models, train_models
from src.preprocessing import clean_data, save_processed_data
from src.utils import ensure_project_dirs

st.set_page_config(
    page_title="Stock Anomaly AI",
    page_icon="📈",
    layout="wide",
)

POPULAR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "V", "NFLX",
]

MODEL_OPTIONS = {
    "Isolation Forest": "Iso_Anomaly",
    "One-Class SVM": "SVM_Anomaly",
}


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def run_pipeline(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Run full ML pipeline via src modules."""
    raw = download_stock_data(symbol=symbol, start=start, end=end, save=True)
    cleaned = clean_data(raw, reset_index=True)
    save_processed_data(cleaned)
    featured = engineer_features(cleaned)
    features = extract_feature_matrix(featured).values

    scaler, iso_model, svm_model, _ = train_models(features)
    save_models(scaler, iso_model, svm_model)

    results, _ = run_detection(
        data=featured,
        features=features,
        scaler=scaler,
        iso_model=iso_model,
        svm_model=svm_model,
        save=True,
    )
    return results


def build_price_chart(
    data: pd.DataFrame,
    anomaly_col: str,
    symbol: str,
    model_name: str,
) -> go.Figure:
    """Close price line with red anomaly markers."""
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    normal = df[df[anomaly_col] == 0]
    anomalies = df[df[anomaly_col] == 1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=normal["Date"],
            y=normal["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#4da3ff", width=2),
        )
    )
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["Date"],
                y=anomalies["Close"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="red", size=10, symbol="circle"),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title=f"{symbol} - {model_name} Anomaly Detection",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_volatility_chart(data: pd.DataFrame) -> go.Figure:
    """Rolling volatility over time."""
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Volatility"],
            mode="lines",
            name="Volatility (10d)",
            line=dict(color="#b794f4", width=2),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Rolling Volatility",
        xaxis_title="Date",
        yaxis_title="Std Dev",
        height=300,
        showlegend=False,
    )
    return fig


def init_session_state() -> None:
    if "results" not in st.session_state:
        st.session_state.results = None
    if "symbol" not in st.session_state:
        st.session_state.symbol = DEFAULT_SYMBOL
    if "summary" not in st.session_state:
        st.session_state.summary = None


def main() -> None:
    ensure_project_dirs()
    init_session_state()

    st.title("Stock Anomaly AI")
    st.caption(
        "Unsupervised anomaly detection | Isolation Forest & One-Class SVM | yFinance"
    )

    with st.sidebar:
        st.header("Controls")

        stock_choice = st.selectbox(
            "Stock",
            options=POPULAR_STOCKS,
            index=POPULAR_STOCKS.index(DEFAULT_SYMBOL)
            if DEFAULT_SYMBOL in POPULAR_STOCKS
            else 0,
        )
        custom_symbol = st.text_input(
            "Custom ticker (optional)",
            placeholder="e.g. AMD",
        ).strip().upper()
        symbol = custom_symbol if custom_symbol else stock_choice

        default_start = parse_date(DEFAULT_START_DATE)
        default_end = parse_date(DEFAULT_END_DATE)

        date_range = st.date_input(
            "Date range",
            value=(default_start, default_end),
            min_value=date(2000, 1, 1),
            max_value=date.today(),
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = default_start, default_end

        model_name = st.radio(
            "Detection model",
            options=list(MODEL_OPTIONS.keys()),
        )
        anomaly_col = MODEL_OPTIONS[model_name]

        run_btn = st.button("Run analysis", type="primary", use_container_width=True)

    if run_btn:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()

        with st.spinner(f"Running pipeline for {symbol} ({start_str} to {end_str})..."):
            try:
                results = run_pipeline(symbol=symbol, start=start_str, end=end_str)
                st.session_state.results = results
                st.session_state.symbol = symbol
                st.session_state.summary = summarize_anomalies(results)
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.stop()

        st.success(f"Analysis complete for {symbol}.")

    results = st.session_state.results
    symbol_display = st.session_state.symbol

    if results is None:
        st.info(
            "Select a stock and date range in the sidebar, then click **Run analysis**."
        )
        st.stop()

    anomaly_col = MODEL_OPTIONS[model_name]
    total_anomalies = int(results[anomaly_col].sum())
    avg_volatility = float(results["Volatility"].mean())

    st.subheader("Key metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total anomalies", total_anomalies)
    c2.metric("Selected stock", symbol_display)
    c3.metric("Avg. volatility", f"{avg_volatility:.4f}")
    c4.metric("Data points", len(results))

    st.subheader("Stock price & anomalies")
    st.plotly_chart(
        build_price_chart(results, anomaly_col, symbol_display, model_name),
        use_container_width=True,
    )

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Volatility")
        st.plotly_chart(build_volatility_chart(results), use_container_width=True)

    with right:
        st.subheader("Detection summary")
        summary = st.session_state.summary or summarize_anomalies(results)
        st.write(f"**Isolation Forest:** {summary['isolation_forest_anomalies']}")
        st.write(f"**One-Class SVM:** {summary['one_class_svm_anomalies']}")
        st.write(f"**Total rows:** {summary['total_points']}")
        st.write(f"**Active model:** {model_name}")

        st.download_button(
            label="Download results CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name=f"{symbol_display}_anomaly_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Preview results"):
        cols = [
            "Date", "Open", "High", "Low", "Close", "Volume",
            "Daily_Return", "Volatility", "Iso_Anomaly", "SVM_Anomaly",
        ]
        show = [c for c in cols if c in results.columns]
        st.dataframe(
            results[show].sort_values("Date", ascending=False).head(50),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
