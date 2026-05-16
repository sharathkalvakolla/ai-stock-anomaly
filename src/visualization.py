"""
Visualization — plot stock price with anomaly markers.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def plot_price_with_anomalies(
    data: pd.DataFrame,
    anomaly_column: str,
    title: str,
    anomaly_label: str,
    anomaly_color: str = "red",
    price_col: str = "Close",
    date_col: str = "Date",
    figsize: tuple = (15, 7),
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot closing price with scatter points for detected anomalies.

    Args:
        data: Results DataFrame with price and anomaly flag columns.
        anomaly_column: Column name for binary anomaly flag (0/1).
        title: Chart title.
        anomaly_label: Legend label for anomaly points.
        anomaly_color: Scatter color for anomalies.
        price_col: Price column to plot.
        date_col: Date column for x-axis.
        figsize: Figure size (width, height).
        save_path: If set, save figure to this path.
        show: Call plt.show() when True.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(data[date_col], data[price_col], label="Stock Price", color="blue")

    anomalies = data[data[anomaly_column] == 1]
    ax.scatter(
        anomalies[date_col],
        anomalies[price_col],
        color=anomaly_color,
        label=anomaly_label,
        zorder=5,
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)

    if show:
        plt.show()

    return fig


def plot_isolation_forest_results(
    data: pd.DataFrame,
    save: bool = False,
    show: bool = True,
) -> plt.Figure:
    """Plot anomalies flagged by Isolation Forest."""
    save_path = (OUTPUTS_DIR / "iso_anomaly_chart.png") if save else None
    return plot_price_with_anomalies(
        data=data,
        anomaly_column="Iso_Anomaly",
        title="Stock Price Anomaly Detection — Isolation Forest",
        anomaly_label="Isolation Forest Anomaly",
        anomaly_color="red",
        save_path=save_path,
        show=show,
    )


def plot_svm_results(
    data: pd.DataFrame,
    save: bool = False,
    show: bool = True,
) -> plt.Figure:
    """Plot anomalies flagged by One-Class SVM."""
    save_path = (OUTPUTS_DIR / "svm_anomaly_chart.png") if save else None
    return plot_price_with_anomalies(
        data=data,
        anomaly_column="SVM_Anomaly",
        title="Stock Price Anomaly Detection — One-Class SVM",
        anomaly_label="SVM Anomaly",
        anomaly_color="orange",
        save_path=save_path,
        show=show,
    )


def generate_all_plots(
    data: pd.DataFrame,
    save: bool = True,
    show: bool = True,
) -> None:
    """Generate both Isolation Forest and SVM visualizations."""
    plot_isolation_forest_results(data, save=save, show=show)
    plot_svm_results(data, save=save, show=show)
