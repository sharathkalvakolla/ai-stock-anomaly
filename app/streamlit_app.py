"""
Stock Anomaly AI - Professional Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtesting import run_backtest
from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    FEATURE_COLUMNS,
    SPLIT_COLUMN,
    TRAIN_RATIO,
)
from src.explainer import build_ai_explanations
from src.pipeline import run_temporal_pipeline
from src.visualization import (
    plot_anomaly_overlap,
    plot_candlestick_with_anomalies,
    plot_feature_distributions,
    plot_price_with_anomalies_plotly,
    plot_volatility,
    plot_technical_indicators,
    plot_feature_correlation_heatmap,
    plot_correlation_heatmap,
    plot_anomaly_distribution,
)

st.set_page_config(page_title="Stock Anomaly AI", layout="wide", page_icon="📈")
st.title("📈 AI Stock Anomaly Detection")
st.markdown("**Temporal Train/Test Split • Unsupervised ML • College Project Dashboard**")


def _normalize_date_input(date_selection):
    if isinstance(date_selection, tuple) and len(date_selection) == 2:
        return date_selection[0], date_selection[1]
    if hasattr(date_selection, "start") and hasattr(date_selection, "stop"):
        return date_selection.start, date_selection.stop
    return date_selection, date_selection


@st.cache_data(show_spinner=False)
def cached_pipeline(symbol, start, end, train_ratio):
    return run_temporal_pipeline(
        symbol=symbol,
        start=str(start),
        end=str(end),
        train_ratio=train_ratio,
        force_retrain=False,
    )


def _get_split_mask(df: pd.DataFrame) -> pd.Series:
    if SPLIT_COLUMN in df.columns:
        return df[SPLIT_COLUMN].astype(str).str.lower() == "test"
    return pd.Series(False, index=df.index)


def _flag_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name in df.columns:
        return df[column_name] == 1
    return pd.Series(False, index=df.index)


def _build_history_entry(symbol, start, end, train_ratio, summary):
    return {
        "symbol": symbol,
        "start": str(start),
        "end": str(end),
        "train_ratio": float(train_ratio),
        "iso_anomalies": int(summary.get("isolation_forest_anomalies", 0)),
        "svm_anomalies": int(summary.get("one_class_svm_anomalies", 0)),
        "both_models": int(summary.get("both_models_anomalies", 0)),
        "test_points": int(summary.get("test_points", 0)),
    }


def _safe_plot(chart_or_fig, fallback_message: str, *args, **kwargs):
    try:
        fig = chart_or_fig(*args, **kwargs) if callable(chart_or_fig) else chart_or_fig
        if fig is None:
            raise ValueError("No figure was returned.")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"{fallback_message} ({exc})")
        st.warning("This chart failed to render, but the dashboard remains active.")


with st.sidebar.form("analysis_form"):
    st.header("🔧 Analysis Configuration")

    symbol = st.selectbox(
        "Choose a Stock Symbol",
        [DEFAULT_SYMBOL, "MSFT", "GOOGL", "AMZN", "TSLA", "AMD", "NVDA"],
        index=0,
    )
    custom = st.text_input("Custom Ticker (e.g. RELIANCE.NS)").strip().upper()
    if custom:
        symbol = custom

    default_start = pd.to_datetime(DEFAULT_START_DATE).date()
    default_end = pd.to_datetime(DEFAULT_END_DATE).date()
    selected_dates = st.date_input(
        "Date Range",
        value=(default_start, default_end),
        min_value=pd.Timestamp("2000-01-01").date(),
        max_value=pd.Timestamp("today").date(),
    )
    start_date, end_date = _normalize_date_input(selected_dates)

    train_ratio = st.slider("Train/Test Split Ratio", 0.6, 0.9, TRAIN_RATIO, 0.05)
    force_retrain = st.checkbox("Force Retrain Models", False)

    submitted = st.form_submit_button("🚀 Run Analysis")

if submitted:
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
    else:
        st.session_state["run"] = True
        st.session_state["symbol"] = symbol
        st.session_state["start"] = start_date
        st.session_state["end"] = end_date
        st.session_state["train_ratio"] = train_ratio
        st.session_state["force_retrain"] = force_retrain

if "run" not in st.session_state:
    st.info("👈 Configure the analysis in the sidebar and click **Run Analysis** to start.")
    st.stop()

try:
    if st.session_state.get("force_retrain", False):
        results, summary, split_info = run_temporal_pipeline(
            symbol=st.session_state["symbol"],
            start=str(st.session_state["start"]),
            end=str(st.session_state["end"]),
            train_ratio=st.session_state["train_ratio"],
            force_retrain=True,
        )
    else:
        results, summary, split_info = cached_pipeline(
            st.session_state["symbol"],
            st.session_state["start"],
            st.session_state["end"],
            st.session_state["train_ratio"],
        )

    if results.empty:
        st.warning("The pipeline completed successfully, but no results were produced.")
        st.stop()

    st.success(f"✅ Analysis Complete for **{st.session_state['symbol']}**")

    history = st.session_state.get("history", [])
    new_entry = _build_history_entry(
        st.session_state["symbol"],
        st.session_state["start"],
        st.session_state["end"],
        st.session_state["train_ratio"],
        summary,
    )
    if not history or history[-1] != new_entry:
        history.append(new_entry)
        st.session_state["history"] = history

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ISO Anomalies", summary.get("isolation_forest_anomalies", 0))
    c2.metric("SVM Anomalies", summary.get("one_class_svm_anomalies", 0))
    c3.metric("Both Models Agree", summary.get("both_models_anomalies", 0))
    c4.metric("Test Period", f"{summary.get('test_points', 0)} days")

    tabs = st.tabs(
        [
            "📊 Price & Candlestick",
            "🔄 Model Comparison",
            "📈 Features",
            "🔍 Explainability",
            "🤖 AI Explainer",
            "📉 Backtesting",
            "🌐 Multi-Stock",
            "📉 Technical Indicators",
        ]
    )

    with tabs[0]:
        st.subheader("💰 Price & Candlestick")
        st.markdown("Beautiful candlestick chart with volume and anomaly markers")
        _safe_plot(
            lambda: plot_candlestick_with_anomalies(results, symbol=st.session_state["symbol"]),
            "Unable to render candlestick chart.",
        )

        # Add volume information
        if "Volume" in results.columns:
            avg_vol = results["Volume"].mean()
            max_vol = results["Volume"].max()
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Volume", f"{avg_vol:,.0f}")
            col2.metric("Peak Volume", f"{max_vol:,.0f}")
            col3.metric("Price Range", f"${results['Close'].min():.2f} - ${results['Close'].max():.2f}")

        # Anomalies table
        anomaly_rows = results.loc[
            _flag_series(results, "Iso_Anomaly") | _flag_series(results, "SVM_Anomaly")
        ]
        if not anomaly_rows.empty:
            st.markdown("---")
            st.markdown("**🚨 Recent Detected Anomalies**")
            display_cols = ["Date", "Close", "Iso_Anomaly", "SVM_Anomaly"]
            if "Iso_Score" in anomaly_rows.columns:
                display_cols.append("Iso_Score")
            if "SVM_Score" in anomaly_rows.columns:
                display_cols.append("SVM_Score")
            st.dataframe(
                anomaly_rows[[col for col in display_cols if col in anomaly_rows.columns]].tail(15),
                use_container_width=True,
            )
        else:
            st.info("✅ No anomalies detected in this period.")

    with tabs[1]:
        st.subheader("🔬 Model Comparison")
        st.markdown("Side-by-side anomaly detection from both models with overlap analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Isolation Forest**")
            _safe_plot(
                lambda: plot_price_with_anomalies_plotly(
                    results,
                    "Iso_Anomaly",
                    st.session_state["symbol"],
                    "Isolation Forest",
                ),
                "Unable to render Isolation Forest comparison.",
            )
        with col2:
            st.markdown("**One-Class SVM**")
            _safe_plot(
                lambda: plot_price_with_anomalies_plotly(
                    results,
                    "SVM_Anomaly",
                    st.session_state["symbol"],
                    "One-Class SVM",
                ),
                "Unable to render One-Class SVM comparison.",
            )
        
        st.markdown("---")
        st.markdown("**Anomaly Detection Overlap**")
        _safe_plot(lambda: plot_anomaly_overlap(summary), "Unable to render anomaly overlap summary.")
        
        # Statistics
        st.markdown("---")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("ISO Forest Only", summary.get("iso_only_anomalies", 0))
        stat_col2.metric("SVM Only", summary.get("svm_only_anomalies", 0))
        stat_col3.metric("Both Agree", summary.get("both_models_anomalies", 0))

    with tabs[2]:
        st.subheader("🎯 Feature Analysis")
        st.markdown("Feature distributions and correlation analysis")
        
        # Feature distributions
        st.markdown("**Distribution: Train vs Test**")
        _safe_plot(
            lambda: plot_feature_distributions(results, feature_columns=FEATURE_COLUMNS),
            "Unable to render feature distributions.",
        )
        
        st.markdown("---")
        
        # Feature correlation heatmap
        st.markdown("**Feature Correlation Matrix**")
        _safe_plot(
            lambda: plot_feature_correlation_heatmap(results),
            "Unable to render feature correlation heatmap.",
        )
        
        st.markdown("---")
        
        # Volatility indicators
        st.markdown("**Volatility & Technical Indicators**")
        _safe_plot(
            lambda: plot_volatility(results, st.session_state["symbol"]),
            "Unable to render volatility indicators.",
        )

    with tabs[3]:
        st.subheader("🔎 Test Set Data")
        st.markdown("Sample of test period records with predictions")
        test_mask = _get_split_mask(results)
        test_data = results.loc[test_mask]
        if not test_data.empty:
            st.markdown(
                f"**Showing {min(20, len(test_data))} of {len(test_data)} test-period records**"
            )
            display_cols = ["Date", "Close", "Iso_Anomaly", "SVM_Anomaly"]
            if "Iso_Score" in test_data.columns:
                display_cols.append("Iso_Score")
            if "SVM_Score" in test_data.columns:
                display_cols.append("SVM_Score")
            
            st.dataframe(
                test_data.loc[:, [col for col in display_cols if col in test_data.columns]].tail(20),
                use_container_width=True,
            )
            
            # Statistics
            st.markdown("---")
            iso_anom = (test_data["Iso_Anomaly"] == 1).sum()
            svm_anom = (test_data["SVM_Anomaly"] == 1).sum()
            both_anom = ((test_data["Iso_Anomaly"] == 1) & (test_data["SVM_Anomaly"] == 1)).sum()
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric("Test Records", len(test_data))
            stat_col2.metric("ISO Anomalies", iso_anom)
            stat_col3.metric("SVM Anomalies", svm_anom)
            stat_col4.metric("Agreement", both_anom)
        else:
            st.info("No test set records available.")

    with tabs[4]:
        st.subheader("🧠 AI Explainer")
        st.markdown("Plain English explanations of detected anomalies")
        train_df = (
            results.loc[results[SPLIT_COLUMN].astype(str).str.lower() == "train"]
            if SPLIT_COLUMN in results.columns
            else results.iloc[:0]
        )
        explanation_df = build_ai_explanations(results, train_df)
        if not explanation_df.empty:
            st.info(f"Found {len(explanation_df)} anomalies with explanations")
            for idx, (_, row) in enumerate(explanation_df.head(5).iterrows()):
                anomaly_date = pd.to_datetime(row['Date']).date()
                anomaly_type = row.get('Anomaly_Type', 'Unknown')
                with st.expander(f"📍 {anomaly_date} — {anomaly_type}", expanded=(idx==0)):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(row.get("Plain_English_Explanation", "No explanation available."))
                    with col2:
                        if "Iso_Score" in row:
                            st.metric("ISO Score", f"{row['Iso_Score']:.2f}")
                        if "SVM_Score" in row:
                            st.metric("SVM Score", f"{row['SVM_Score']:.2f}")
        else:
            st.info(
                "No AI explanations available. Ensure the selected date range produces test-period anomalies."
            )

    with tabs[5]:
        st.subheader("📊 Backtesting Results")
        st.markdown("Strategy performance vs Buy & Hold on test period")
        test_mask = _get_split_mask(results)
        test_df = results.loc[test_mask].copy()
        if not test_df.empty:
            equity_df, backtest_metrics, trades_df = run_backtest(test_df, anomaly_col="Iso_Anomaly")
            
            # Metrics
            metric_cols = st.columns(4)
            metric_cols[0].metric(
                "Strategy Return",
                f"{backtest_metrics['Strategy Total Return %']:.2f}%",
                delta=f"{backtest_metrics['Strategy Total Return %'] - backtest_metrics['BuyHold Total Return %']:.2f}%"
            )
            metric_cols[1].metric("Buy & Hold Return", f"{backtest_metrics['BuyHold Total Return %']:.2f}%")
            metric_cols[2].metric("Win Rate", f"{backtest_metrics['Win Rate %']:.1f}%")
            metric_cols[3].metric("Total Trades", backtest_metrics.get('Total Trades', 0))

            st.markdown("---")
            st.markdown("**Equity Curve**")
            _safe_plot(
                lambda: plot_price_with_anomalies_plotly(
                    equity_df.rename(columns={"Strategy": "Close"}),
                    anomaly_col="",
                    symbol=st.session_state["symbol"],
                    series_name="Strategy Equity",
                ),
                "Unable to render backtest equity curve.",
            )

            st.markdown("---")
            st.markdown("**Detailed Performance**")
            st.dataframe(backtest_metrics, use_container_width=True)
            
            st.markdown("**Recent Trades**")
            st.dataframe(trades_df.tail(15), use_container_width=True)
        else:
            st.info("Backtesting requires a test set. Try a larger date range or adjust the split ratio.")

    with tabs[6]:
        st.subheader("🌍 Multi-Stock Analysis")
        st.markdown("Cross-stock anomaly and returns correlation analysis")
        
        history = st.session_state.get("history", [])
        if history:
            history_df = pd.DataFrame(history)
            st.markdown(f"**Analysis History ({len(history)} runs)**")
            st.dataframe(
                history_df.drop(columns=["symbol"]) if "symbol" in history_df else history_df,
                use_container_width=True,
            )
            
            # Anomaly comparison chart
            st.markdown("---")
            st.markdown("**Anomaly Counts by Symbol**")
            comparison_fig = go.Figure(
                data=[
                    go.Bar(
                        name="ISO Anomalies",
                        x=history_df["symbol"],
                        y=history_df["iso_anomalies"],
                        marker_color="#fb7185",
                    ),
                    go.Bar(
                        name="SVM Anomalies",
                        x=history_df["symbol"],
                        y=history_df["svm_anomalies"],
                        marker_color="#60a5fa",
                    ),
                    go.Bar(
                        name="Both Agree",
                        x=history_df["symbol"],
                        y=history_df["both_models"],
                        marker_color="#facc15",
                    ),
                ]
            )
            comparison_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                barmode="group",
                title="Anomaly Counts Across Symbols",
                xaxis_title="Symbol",
                yaxis_title="Anomaly Count",
                height=520,
                margin=dict(t=60, b=40, l=60, r=30),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )
            _safe_plot(
                comparison_fig,
                "Unable to render multi-stock comparison.",
            )
            
            # Returns correlation (if we have multiple stocks)
            if len(history_df) > 1:
                st.markdown("---")
                st.markdown("**Returns Correlation Matrix**")
                st.info("💡 Add more stocks to enable returns correlation analysis. Run analysis for different symbols.")
        else:
            st.info(
                "👉 Run the analysis for multiple symbols to build a comprehensive comparison. Try: AAPL, MSFT, GOOGL"
            )

    with tabs[7]:
        st.subheader("📈 Technical Analysis")
        st.markdown("RSI, MACD, Bollinger Bands, and Volume analysis")
        if callable(plot_technical_indicators):
            _safe_plot(
                lambda: plot_technical_indicators(results, st.session_state["symbol"]),
                "Unable to render technical indicators.",
            )
        else:
            st.error("Technical indicators module not available. Please refresh or check the visualization module.")

    st.markdown("---")

    st.download_button(
        label="📥 Download Full Results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=f"{st.session_state['symbol']}_anomaly_results.csv",
        mime="text/csv",
    )

except Exception as error:
    st.error(f"Something went wrong during the analysis: {error}")
    st.info("Try enabling Force Retrain Models or using a shorter date range.")

st.caption("Major Project | Made with ❤️ using AI tools | Not Financial Advice")