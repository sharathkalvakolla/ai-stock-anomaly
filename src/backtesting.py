from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    return float(drawdown.min()) * 100.0


def _sharpe(returns: pd.Series) -> float:
    if returns.std(ddof=0) == 0:
        return 0.0
    return float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252))


def run_backtest(test_df: pd.DataFrame, anomaly_col: str, reentry_days: int = 5) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    df = test_df.copy().sort_values("Date").reset_index(drop=True)
    prices = df["Close"].astype(float).values
    dates = pd.to_datetime(df["Date"])
    start_cash = 10000.0

    # Buy-and-hold
    bh_shares = start_cash / prices[0]
    bh_equity = bh_shares * prices

    # Strategy: exit on anomaly, re-enter after N days
    in_market = True
    cash = 0.0
    shares = start_cash / prices[0]
    cooldown = 0
    trade_log: List[Dict[str, float | str]] = [{"Date": str(dates.iloc[0].date()), "Action": "BUY", "Price": prices[0], "Portfolio Value": start_cash, "Return": 0.0}]
    strategy_equity: List[float] = []
    wins = 0
    exit_count = 0

    for i in range(len(df)):
        price = prices[i]
        is_anomaly = int(df.loc[i, anomaly_col]) == 1
        if in_market and is_anomaly:
            cash = shares * price
            shares = 0.0
            in_market = False
            cooldown = reentry_days
            exit_count += 1
            next_window = prices[i + 1 : min(i + reentry_days + 1, len(prices))]
            if len(next_window) > 0 and min(next_window) < price:
                wins += 1
            trade_log.append({"Date": str(dates.iloc[i].date()), "Action": "SELL", "Price": price, "Portfolio Value": cash, "Return": 0.0})
        elif not in_market:
            cooldown -= 1
            if cooldown <= 0:
                shares = cash / price
                cash = 0.0
                in_market = True
                trade_log.append({"Date": str(dates.iloc[i].date()), "Action": "BUY", "Price": price, "Portfolio Value": shares * price, "Return": 0.0})
        strategy_equity.append(shares * price if in_market else cash)

    strat_equity_series = pd.Series(strategy_equity, index=dates)
    bh_equity_series = pd.Series(bh_equity, index=dates)
    strat_rets = strat_equity_series.pct_change().fillna(0.0)
    bh_rets = bh_equity_series.pct_change().fillna(0.0)

    equity_df = pd.DataFrame(
        {
            "Date": dates,
            "Strategy": strat_equity_series.values,
            "BuyHold": bh_equity_series.values,
        }
    )
    metrics = {
        "Strategy Total Return %": float((strat_equity_series.iloc[-1] / strat_equity_series.iloc[0] - 1.0) * 100.0),
        "BuyHold Total Return %": float((bh_equity_series.iloc[-1] / bh_equity_series.iloc[0] - 1.0) * 100.0),
        "Strategy Max Drawdown %": _max_drawdown(strat_equity_series),
        "BuyHold Max Drawdown %": _max_drawdown(bh_equity_series),
        "Win Rate %": float((wins / exit_count) * 100.0) if exit_count > 0 else 0.0,
        "Strategy Sharpe": _sharpe(strat_rets),
        "BuyHold Sharpe": _sharpe(bh_rets),
        "Trades": int(len(trade_log) - 1),
    }
    trades_df = pd.DataFrame(trade_log)
    return equity_df, metrics, trades_df
