"""
Professional Stock Anomaly Detection Visualization Module
Provides rich, modern Plotly charts with proper error handling and responsive design.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import SPLIT_COLUMN


def _clean_date_column(data: pd.DataFrame) -> pd.DataFrame:
    """Convert Date column to datetime format with validation."""
    df = data.copy()
    if "Date" not in df.columns:
        raise ValueError("The dataset must include a Date column.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().all():
        raise ValueError("Unable to parse any dates in the Date column.")

    return df


def _create_empty_figure(message: str, height: int = 420) -> go.Figure:
    """Create a professional empty figure with centered message."""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        font=dict(size=16, color="#94a3b8"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Segoe UI, -apple-system, sans-serif"),
        height=height,
        margin=dict(t=40, b=40, l=40, r=40),
    )
    return fig


def add_vertical_marker(
    fig: go.Figure,
    x_date,
    line_color: str = "#facc15",
    line_dash: str = "dash",
    label: str | None = None,
) -> None:
    """
    Add a vertical marker line to a figure with optional label.
    Uses proper Plotly references (not deprecated xrefsrc).
    """
    try:
        x = pd.to_datetime(x_date, errors="coerce")
        if pd.isna(x):
            return

        x_str = x.isoformat()

        # Add vertical line shape
        fig.add_shape(
            type="line",
            x0=x_str,
            x1=x_str,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color=line_color, width=2, dash=line_dash),
            layer="below",
        )

        # Add optional label
        if label:
            fig.add_annotation(
                x=x_str,
                y=1.02,
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(size=12, color=line_color),
                align="center",
                bordercolor=line_color,
                borderwidth=1,
                borderpad=4,
                bgcolor="#111827",
                opacity=0.95,
            )
    except Exception as e:
        pass  # Silently ignore marker errors


def plot_candlestick_with_anomalies(
    data,
    anomaly_col: str = "Iso_Anomaly",
    symbol: str = "Stock",
) -> go.Figure:
    """
    Create a professional candlestick chart with anomaly markers and volume.
    """
    try:
        df = _clean_date_column(data)

        required_columns = {"Open", "High", "Low", "Close"}
        if not required_columns.issubset(df.columns):
            return _create_empty_figure("Price data (OHLC) is incomplete.")

        # Create figure with secondary y-axis for volume
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.75, 0.25],
        )

        # Candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
            ),
            row=1,
            col=1,
        )

        # Anomaly markers
        if anomaly_col in df.columns:
            anomalies = df[df[anomaly_col] == 1]
            if not anomalies.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies["Date"],
                        y=anomalies["Close"],
                        mode="markers",
                        name="Anomaly",
                        marker=dict(
                            color="#ef4444",
                            size=12,
                            symbol="x",
                            line=dict(color="#dc2626", width=2),
                        ),
                        hovertemplate="<b>ANOMALY</b><br>%{x|%Y-%m-%d}<br>Close: $%{y:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        # Volume bars
        if "Volume" in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df["Date"],
                    y=df["Volume"],
                    name="Volume",
                    marker_color="#1e40af",
                    opacity=0.3,
                    hovertemplate="%{x|%Y-%m-%d}<br>Volume: %{y:.0f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # Test/Train split marker
        if SPLIT_COLUMN in df.columns:
            test_dates = df.loc[
                df[SPLIT_COLUMN].astype(str).str.lower() == "test", "Date"
            ]
            if not test_dates.empty:
                add_vertical_marker(
                    fig,
                    test_dates.iloc[0],
                    line_color="#facc15",
                    label="Test Start →",
                )

        fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=1)
        fig.update_xaxes(showgrid=False, zeroline=False, row=2, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#334155", row=1, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#334155", row=2, col=1)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Segoe UI, -apple-system, sans-serif"),
            title=dict(text=f"<b>{symbol} Candlestick Chart</b>", x=0.5),
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=720,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0.3)",
            ),
            margin=dict(t=60, b=40, l=60, r=30),
            hovermode="x unified",
        )

        return fig

    except Exception as e:
        return _create_empty_figure(f"Candlestick chart error: {str(e)}")


def plot_price_with_anomalies_plotly(
    data,
    anomaly_col: str,
    symbol: str,
    series_name: str,
) -> go.Figure:
    """
    Create a line chart with price and anomaly markers.
    """
    try:
        df = _clean_date_column(data)

        fig = go.Figure()

        # Close price line
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="#38bdf8", width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
            )
        )

        # Anomaly markers
        if anomaly_col and anomaly_col in df.columns:
            anomalies = df[df[anomaly_col] == 1]
            if not anomalies.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomalies["Date"],
                        y=anomalies["Close"],
                        mode="markers",
                        name=f"{series_name} Anomaly",
                        marker=dict(
                            size=11,
                            color="#ef4444",
                            symbol="triangle-up",
                            line=dict(color="#dc2626", width=2),
                        ),
                        hovertemplate="<b>ANOMALY</b><br>%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
                    )
                )

        # Test/Train split
        if SPLIT_COLUMN in df.columns:
            test_dates = df.loc[
                df[SPLIT_COLUMN].astype(str).str.lower() == "test", "Date"
            ]
            if not test_dates.empty:
                add_vertical_marker(fig, test_dates.iloc[0], line_color="#facc15", label="Test")

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Segoe UI, -apple-system, sans-serif"),
            title=dict(text=f"<b>{symbol} — {series_name}</b>", x=0.5),
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=520,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0.3)",
            ),
            margin=dict(t=60, b=40, l=60, r=30),
            hovermode="x unified",
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#334155")

        return fig

    except Exception as e:
        return _create_empty_figure(f"Price chart error: {str(e)}")


def plot_anomaly_overlap(summary: dict) -> go.Figure:
    """
    Create a bar chart showing anomaly overlap between models.
    """
    try:
        categories = [
            "Isolation Forest Only",
            "One-Class SVM Only",
            "Both Models",
        ]
        values = [
            int(summary.get("iso_only_anomalies", 0)),
            int(summary.get("svm_only_anomalies", 0)),
            int(summary.get("both_models_anomalies", 0)),
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=categories,
                    y=values,
                    marker_color=["#fb7185", "#60a5fa", "#facc15"],
                    text=[f"{v}" for v in values],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Segoe UI, -apple-system, sans-serif"),
            title=dict(text="<b>Anomaly Detection Overlap</b>", x=0.5),
            xaxis_title="Detection Method",
            yaxis_title="Count",
            yaxis=dict(showgrid=True, gridcolor="#334155"),
            height=520,
            margin=dict(t=60, b=40, l=60, r=30),
            showlegend=False,
        )

        return fig

    except Exception as e:
        return _create_empty_figure(f"Anomaly overlap error: {str(e)}")


def plot_feature_distributions(
    data,
    feature_columns: list[str] | None = None,
) -> go.Figure:
    """
    Create histograms for feature distributions with train/test comparison.
    """
    try:
        df = data.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        numeric_cols = [
            column
            for column in df.select_dtypes(include="number").columns
            if column not in ["Iso_Anomaly", "SVM_Anomaly", "Iso_Score", "SVM_Score"]
        ]

        if feature_columns:
            numeric_cols = [
                col for col in feature_columns
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]

        if not numeric_cols:
            return _create_empty_figure("No numeric features available.", height=380)

        feature_cols = numeric_cols[:6]
        rows = (len(feature_cols) + 2) // 3
        cols = min(3, len(feature_cols))

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=feature_cols,
            specs=[[{"type": "histogram"}] * cols for _ in range(rows)],
        )

        split_present = SPLIT_COLUMN in df.columns
        if split_present:
            train_mask = df[SPLIT_COLUMN].astype(str).str.lower() == "train"
            test_mask = df[SPLIT_COLUMN].astype(str).str.lower() == "test"
        else:
            train_mask = pd.Series(False, index=df.index)
            test_mask = pd.Series(False, index=df.index)

        for idx, feature in enumerate(feature_cols):
            row = idx // cols + 1
            col = idx % cols + 1

            if split_present:
                if train_mask.any():
                    fig.add_trace(
                        go.Histogram(
                            x=df.loc[train_mask, feature].dropna(),
                            name="Train",
                            marker_color="#22c55e",
                            opacity=0.65,
                            showlegend=(idx == 0),
                            nbinsx=30,
                        ),
                        row=row,
                        col=col,
                    )
                if test_mask.any():
                    fig.add_trace(
                        go.Histogram(
                            x=df.loc[test_mask, feature].dropna(),
                            name="Test",
                            marker_color="#facc15",
                            opacity=0.65,
                            showlegend=(idx == 0),
                            nbinsx=30,
                        ),
                        row=row,
                        col=col,
                    )
            else:
                fig.add_trace(
                    go.Histogram(
                        x=df[feature].dropna(),
                        name=feature,
                        marker_color="#38bdf8",
                        opacity=0.85,
                        showlegend=False,
                        nbinsx=30,
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Segoe UI, -apple-system, sans-serif"),
            title=dict(text="<b>Feature Distribution: Train vs Test</b>", x=0.5),
            barmode="overlay",
            height=240 * rows + 140,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.0,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0.3)",
            ),
            margin=dict(t=60, b=40, l=60, r=30),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#334155")

        return fig

    except Exception as e:
        return _create_empty_figure(f"Feature distribution error: {str(e)}")


def plot_volatility(data, symbol: str = "Stock") -> go.Figure:
    """
    Create a volatility and technical indicator trends chart.
    """
    try:
        df = _clean_date_column(data)

        volatility_columns = [
            column
            for column in ["Volatility", "Rolling_STD", "BB_Width", "Volume_MA_20"]
            if column in df.columns
        ]

        if not volatility_columns:
            return _create_empty_figure("No volatility indicators available.", height=420)

        fig = go.Figure()
        palette = ["#60a5fa", "#f472b6", "#facc15", "#22c55e"]

        for idx, column in enumerate(volatility_columns):
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df[column],
                    mode="lines",
                    name=column,
                    line=dict(width=2.5, color=palette[idx % len(palette)]),
                    hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}<br>Value: %{y:.3f}<extra></extra>",
                )
            )

        if SPLIT_COLUMN in df.columns:
            test_dates = df.loc[
                df[SPLIT_COLUMN].astype(str).str.lower() == "test", "Date"
            ]
            if not test_dates.empty:
                add_vertical_marker(
                    fig, test_dates.iloc[0], line_color="#facc15", label="Test"
                )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Segoe UI, -apple-system, sans-serif"),
            title=dict(text=f"<b>{symbol} Volatility & Indicators</b>", x=0.5),
            xaxis_title="Date",
            yaxis_title="Value",
            height=520,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0.3)",
            ),
            margin=dict(t=60, b=40, l=60, r=30),
            hovermode="x unified",
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#334155")

        return fig

    except Exception as e:
        return _create_empty_figure(f"Volatility chart error: {str(e)}")


def plot_technical_indicators(data, symbol: str = "Stock") -> go.Figure:
    """
    Create a comprehensive technical indicators subplot with RSI, MACD, and Bollinger Bands.
    """
    try:
        df = _clean_date_column(data)

        if "Close" not in df.columns:
            return _create_empty_figure("Close price data unavailable.")

        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[
                "Price & Bollinger Bands",
                "RSI (14)",
                "MACD",
                "Volume & Indicators",
            ],
            row_heights=[0.35, 0.20, 0.25, 0.20],
        )

        # Row 1: Price and Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                name="Close",
                line=dict(color="#38bdf8", width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["BB_Upper"],
                    name="BB Upper",
                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["BB_Lower"],
                    name="BB Lower",
                    line=dict(color="#10b981", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            if "BB_Middle" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=df["BB_Middle"],
                        name="BB Middle (SMA 20)",
                        line=dict(color="#94a3b8", width=1, dash="dash"),
                        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        # Row 2: RSI
        if "RSI" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["RSI"],
                    name="RSI (14)",
                    line=dict(color="#f59e0b", width=2.5),
                    hovertemplate="%{x|%Y-%m-%d}<br>RSI: %{y:.1f}<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig.add_hline(y=70, line_dash="dash", line_color="#f87171", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=2, col=1)
        else:
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref=f"y{2} domain",
                text="RSI unavailable",
                showarrow=False,
                font=dict(size=13, color="#94a3b8"),
            )

        # Row 3: MACD
        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["MACD"],
                    name="MACD",
                    line=dict(color="#60a5fa", width=2.5),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["MACD_Signal"],
                    name="Signal (EMA 9)",
                    line=dict(color="#fb7185", width=2, dash="dash"),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
                ),
                row=3,
                col=1,
            )
            if "MACD_Hist" in df.columns:
                colors = [
                    "#22c55e" if x >= 0 else "#ef4444"
                    for x in df["MACD_Hist"]
                ]
                fig.add_trace(
                    go.Bar(
                        x=df["Date"],
                        y=df["MACD_Hist"],
                        name="MACD Histogram",
                        marker_color=colors,
                        opacity=0.5,
                        showlegend=False,
                        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
                    ),
                    row=3,
                    col=1,
                )
        else:
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref=f"y{3} domain",
                text="MACD unavailable",
                showarrow=False,
                font=dict(size=13, color="#94a3b8"),
            )

        # Row 4: Volume
        if "Volume" in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df["Date"],
                    y=df["Volume"],
                    name="Volume",
                    marker_color="#1e40af",
                    opacity=0.4,
                    showlegend=False,
                    hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:.0f}<extra></extra>",
                ),
                row=4,
                col=1,
            )

        if "Volume_MA_20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["Volume_MA_20"],
                    name="Volume MA (20)",
                    line=dict(color="#34d399", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f}<extra></extra>",
                ),
                row=4,
                col=1,
            )

        # Add test/train split marker
        if SPLIT_COLUMN in df.columns:
            test_dates = df.loc[
                df[SPLIT_COLUMN].astype(str).str.lower() == "test", "Date"
            ]
            if not test_dates.empty:
                add_vertical_marker(
                    fig, test_dates.iloc[0], line_color="#facc15", label="Test →"
                )

        # Update layout
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#334155")

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Segoe UI, -apple-system, sans-serif"),
            title=dict(text=f"<b>{symbol} Technical Analysis</b>", x=0.5),
            height=980,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.0,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0.3)",
            ),
            margin=dict(t=70, b=50, l=70, r=30),
            hovermode="x unified",
        )

        return fig

    except Exception as e:
        return _create_empty_figure(f"Technical indicators error: {str(e)}")


def plot_feature_correlation_heatmap(data) -> go.Figure:
    """
    Create a professional correlation heatmap for features.
    """
    try:
        df = data.copy()
        if "Date" in df.columns:
            df = df.drop(columns=["Date"])

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] < 2:
            return _create_empty_figure("Insufficient numeric features for correlation.")

        # Calculate correlation
        corr = numeric_df.corr()

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                colorbar=dict(
                    title="Correlation",
                    thickness=15,
                    len=0.7,
                ),
                hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Segoe UI, -apple-system, sans-serif"),
            title=dict(text="<b>Feature Correlation Matrix</b>", x=0.5),
            xaxis_title="Features",
            yaxis_title="Features",
            height=600,
            margin=dict(t=60, b=120, l=120, r=80),
            coloraxis_colorbar_tickfont=dict(color="#cbd5e1"),
        )

        return fig

    except Exception as e:
        return _create_empty_figure(f"Correlation heatmap error: {str(e)}")


def plot_correlation_heatmap(returns_data: dict | pd.DataFrame) -> go.Figure:
    """
    Create a returns correlation heatmap for multi-stock comparison.
    """
    try:
        if isinstance(returns_data, dict):
            # Build DataFrame from dictionary of returns
            if not returns_data:
                return _create_empty_figure("No returns data available.")

            # Align dates and create DataFrame
            min_len = min(len(v) for v in returns_data.values()) if returns_data else 0
            if min_len == 0:
                return _create_empty_figure("Insufficient returns data.")

            df = pd.DataFrame({k: v[:min_len] for k, v in returns_data.items()})
        else:
            df = returns_data.copy()

        if df.shape[0] < 2 or df.shape[1] < 2:
            return _create_empty_figure("Need at least 2 symbols with sufficient data.")

        # Calculate correlation
        corr = df.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text:.2f}",
                textfont={"size": 11},
                colorbar=dict(
                    title="Correlation",
                    thickness=15,
                    len=0.7,
                ),
                hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Segoe UI, -apple-system, sans-serif"),
            title=dict(text="<b>Returns Correlation Matrix</b>", x=0.5),
            xaxis_title="Symbols",
            yaxis_title="Symbols",
            height=600,
            margin=dict(t=60, b=100, l=100, r=80),
            coloraxis_colorbar_tickfont=dict(color="#cbd5e1"),
        )

        return fig

    except Exception as e:
        return _create_empty_figure(f"Returns correlation error: {str(e)}")


def plot_anomaly_distribution(data, anomaly_col: str = "Iso_Anomaly") -> go.Figure:
    """
    Create a heatmap showing anomaly distribution across dates and price levels.
    """
    try:
        df = _clean_date_column(data)

        if anomaly_col not in df.columns or "Close" not in df.columns:
            return _create_empty_figure("Insufficient data for anomaly distribution.")

        # Get anomaly data
        anomalies = df[df[anomaly_col] == 1].copy()

        if anomalies.empty:
            return _create_empty_figure("No anomalies detected in this period.")

        # Create month-day bins for heatmap
        anomalies["Month"] = anomalies["Date"].dt.to_period("M").astype(str)
        anomalies["DayOfWeek"] = anomalies["Date"].dt.day_name()

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        # Pivot to create heatmap data
        heatmap_data = anomalies.groupby(["Month", "DayOfWeek"]).size().reset_index(name="Count")

        # Create pivot for visualization
        pivot_data = heatmap_data.pivot_table(
            index="DayOfWeek", columns="Month", values="Count", fill_value=0
        )

        # Reorder by day of week
        pivot_data = pivot_data.reindex([d for d in day_order if d in pivot_data.index])

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns.astype(str),
                y=pivot_data.index,
                colorscale="Hot",
                text=pivot_data.values,
                texttemplate="%{text}",
                textfont={"size": 11},
                colorbar=dict(title="Anomalies", thickness=15, len=0.7),
                hovertemplate="<b>%{y}</b> in <b>%{x}</b><br>Count: %{z}<extra></extra>",
            )
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Segoe UI, -apple-system, sans-serif"),
            title=dict(text="<b>Anomaly Distribution by Time</b>", x=0.5),
            xaxis_title="Month",
            yaxis_title="Day of Week",
            height=500,
            margin=dict(t=60, b=80, l=100, r=80),
            coloraxis_colorbar_tickfont=dict(color="#cbd5e1"),
        )

        return fig

    except Exception as e:
        return _create_empty_figure(f"Anomaly distribution error: {str(e)}")