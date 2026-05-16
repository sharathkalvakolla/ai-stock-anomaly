"""
Stock Anomaly AI — Streamlit dashboard (dark theme, Plotly).

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
# Project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anomaly_detection import run_detection, summarize_anomalies
from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    OUTPUTS_DIR,
    RESULTS_CSV_NAME,
)
from src.data_collection import download_stock_data
from src.feature_engineering import engineer_features, extract_feature_matrix
from src.model_training import save_models, train_models
from src.preprocessing import clean_data, save_processed_data
from src.utils import ensure_project_dirs

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Stock Anomaly AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark theme CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(160deg, #0f1117 0%, #1a1d29 45%, #12141c 100%);
        color: #e8eaed;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161922 0%, #0d0f14 100%);
        border-right: 1px solid #2a2f3d;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #f0f2f5;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e2433 0%, #252b3b 100%);
        border: 1px solid #3d4558;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
    }
    div[data-testid="stMetric"] label {
        color: #9aa0b0 !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.75rem;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #ff6b6b !important;
    }

    /* Headers */
    h1 {
        background: linear-gradient(90deg, #6ee7ff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #8b93a7;
        font-size: 1.05rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
    }

    /* Download button */
    .stDownloadButton > button {
        background: #252b3b;
        border: 1px solid #4f5d75;
        color: #e8eaed;
        border-radius: 10px;
    }

    /* Dividers & captions */
    hr {
        border-color: #2a2f3d;
    }
    .stCaption {
        color: #8b93a7;
    }

    /* Hide Streamlit footer branding clutter */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POPULAR_STOCKS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "JPM",
    "V",
    "NFLX",
]

MODEL_OPTIONS = {
    "Isolation Forest": "Iso_Anomaly",
    "One-Class SVM": "SVM_Anomaly",
}

PLOTLY_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#c9cdd4"),
    margin=dict(l=48, r=24, t=56, b=48),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(gridcolor="#2a2f3d", showgrid=True),
    yaxis=dict(gridcolor="#2a2f3d", showgrid=True),
)


def _parse_default_date(value: str) -> date:
    """Parse YYYY-MM-DD config string to date."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def run_pipeline(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Execute full ML pipeline using src modules.

    Returns:
        Results DataFrame with anomaly flags and engineered features.
    """
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
    """
    Interactive Plotly chart: close price line + red anomaly markers.
    """
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
            line=dict(color="#60a5fa", width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Close: $%{y:.2f}<extra></extra>",
        )
    )

    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["Date"],
                y=anomalies["Close"],
                mode="markers",
                name="Anomaly",
                marker=dict(
                    color="#ef4444",
                    size=11,
                    symbol="circle",
                    line=dict(color="#fca5a5", width=1.5),
                ),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Close: $%{y:.2f}<br>"
                    "Status: Anomaly<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title=dict(
            text=f"{symbol} — {model_name} Anomaly Detection",
            font=dict(size=20, color="#f0f2f5"),
            x=0,
            xanchor="left",
        ),
        height=520,
        yaxis_title="Price (USD)",
        xaxis_title="Date",
    )

    return fig


def build_volatility_chart(data: pd.DataFrame) -> go.Figure:
    """Secondary chart: rolling volatility over time."""
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Volatility"],
            mode="lines",
            name="Volatility (10d)",
            line=dict(color="#a78bfa", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(167, 139, 250, 0.12)",
        )
    )
    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title=dict(text="Rolling Volatility", font=dict(size=16), x=0, xanchor="left"),
        height=280,
        yaxis_title="Std Dev",
        showlegend=False,
    )
    return fig


def init_session_state() -> None:
    """Initialize Streamlit session keys."""
    defaults = {
        "results": None,
        "symbol": DEFAULT_SYMBOL,
        "summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_project_dirs()
    init_session_state()

    # Header
    st.title("Stock Anomaly AI")
    st.markdown(
        '<p class="hero-subtitle">'
        "Unsupervised anomaly detection on market data &nbsp;·&nbsp; "
        "Isolation Forest & One-Class SVM &nbsp;·&nbsp; yFinance"
        "</p>",
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Controls")
        st.markdown("---")

        stock_choice = st.selectbox(
            "Stock",
            options=POPULAR_STOCKS,
            index=POPULAR_STOCKS.index(DEFAULT_SYMBOL)
            if DEFAULT_SYMBOL in POPULAR_STOCKS
            else 0,
            help="Select a ticker or enter a custom symbol below.",
        )
        custom_symbol = st.text_input(
            "Custom ticker (optional)",
            placeholder="e.g. AMD",
            help="Overrides the dropdown when filled.",
        ).strip().upper()

        symbol = custom_symbol if custom_symbol else stock_choice

        default_start = _parse_default_date(DEFAULT_START_DATE)
        default_end = _parse_default_date(DEFAULT_END_DATE)

        date_range = st.date_input(
            "Date range",
            value=(default_start, default_end),
            min_value=date(2000, 1, 1),
            max_value=date.today(),
            help="Analysis window for historical data.",
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = default_start
            end_date = default_end

        st.markdown("---")
        model_name = st.radio(
            "Detection model",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
            help="Choose which model's anomaly flags to visualize.",
        )
        anomaly_col = MODEL_OPTIONS[model_name]

        st.markdown("---")
        run_btn = st.button("Run analysis", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption("Pipeline: download > preprocess > features > train > detect")

    # Run pipeline
    if run_btn:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        if start_date >= end_date:
            st.error("Start date must be before end date.")
            return

        with st.spinner(f"Running pipeline for {symbol} ({start_str} to {end_str})..."):
            try:
                results = run_pipeline(symbol=symbol, start=start_str, end=end_str)
                st.session_state.results = results
                st.session_state.symbol = symbol
                st.session_state.summary = summarize_anomalies(results)
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                return

        st.success(f"Analysis complete for **{symbol}**.")

    results: pd.DataFrame | None = st.session_state.results
    symbol_display: str = st.session_state.symbol

    if results is None:
        st.markdown("---")
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.info(
                "Configure **stock**, **date range**, and **model** in the sidebar, "
                "then click **Run analysis** to launch the ML pipeline."
            )
        with col_right:
            st.markdown(
                """
                **What you get**
                - Interactive price chart with anomaly highlights
                - Key metrics at a glance
                - CSV export of full results
                """
            )
        return

    anomaly_col = MODEL_OPTIONS[model_name]
    total_anomalies = int(results[anomaly_col].sum())
    avg_volatility = float(results["Volatility"].mean()) if "Volatility" in results.columns else 0.0

    # Metrics row
    st.markdown("### Key metrics")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            label="Total anomalies",
            value=total_anomalies,
            delta=f"{100 * total_anomalies / len(results):.1f}% of data",
            delta_color="inverse",
        )
    with m2:
        st.metric(label="Selected stock", value=symbol_display)
    with m3:
        st.metric(label="Avg. volatility", value=f"{avg_volatility:.4f}")
    with m4:
        st.metric(label="Data points", value=len(results))

    st.markdown("---")

    # Main chart
    st.plotly_chart(
        build_price_chart(results, anomaly_col, symbol_display, model_name),
        use_container_width=True,
    )

    # Volatility + summary row
    chart_col, info_col = st.columns([2, 1])

    with chart_col:
        st.plotly_chart(build_volatility_chart(results), use_container_width=True)

    with info_col:
        st.markdown("### Detection summary")
        summary = st.session_state.summary or summarize_anomalies(results)
        st.markdown(
            f"""
            | Model | Anomalies |
            |-------|-----------|
            | Isolation Forest | **{summary['isolation_forest_anomalies']}** |
            | One-Class SVM | **{summary['one_class_svm_anomalies']}** |
            | Total rows | **{summary['total_points']}** |
            """
        )
        st.markdown(f"**Active model:** {model_name}")

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results CSV",
            data=csv_bytes,
            file_name=f"{symbol_display}_anomaly_results.csv",
            mime="text/csv",
            use_container_width=True,
            help="Export full dataset including features and anomaly flags.",
        )

    # Data preview
    with st.expander("Preview results table", expanded=False):
        display_cols = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Daily_Return",
            "Volatility",
            "Iso_Anomaly",
            "SVM_Anomaly",
        ]
        available = [c for c in display_cols if c in results.columns]
        st.dataframe(
            results[available].sort_values("Date", ascending=False).head(50),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
