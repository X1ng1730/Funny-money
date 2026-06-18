from __future__ import annotations

import pandas as pd


def calculate_backtest_metrics(trades: pd.DataFrame, initial_equity: float = 100000.0) -> dict:
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_holding_days": 0.0,
        }
    r = pd.to_numeric(trades["r_multiple"], errors="coerce").fillna(0)
    wins = r[r > 0]
    losses = r[r < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else (float("inf") if wins.sum() > 0 else 0.0)
    equity = pd.to_numeric(trades.get("equity_after", pd.Series(dtype=float)), errors="coerce")
    if equity.empty or equity.isna().all():
        equity = initial_equity + pd.to_numeric(trades["pnl_dollars"], errors="coerce").fillna(0).cumsum()
    running_max = equity.cummax()
    drawdown = ((equity - running_max) / running_max).fillna(0)
    total_return = ((equity.iloc[-1] - initial_equity) / initial_equity) * 100 if initial_equity else 0
    holding = pd.to_numeric(trades.get("holding_days", pd.Series(dtype=float)), errors="coerce").fillna(0)
    return {
        "total_trades": int(len(trades)),
        "win_rate": round(float((r > 0).mean() * 100), 2),
        "avg_r": round(float(r.mean()), 3),
        "median_r": round(float(r.median()), 3),
        "expectancy_r": round(float(r.mean()), 3),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else float("inf"),
        "total_return_pct": round(float(total_return), 2),
        "max_drawdown_pct": round(float(drawdown.min() * 100), 2),
        "avg_holding_days": round(float(holding.mean()), 1),
        "best_trade_r": round(float(r.max()), 2),
        "worst_trade_r": round(float(r.min()), 2),
        "target_exits": int(trades["exit_reason"].fillna("").str.contains("Target").sum()) if "exit_reason" in trades else 0,
        "stop_exits": int(trades["exit_reason"].fillna("").str.contains("Stop").sum()) if "exit_reason" in trades else 0,
        "time_exits": int(trades["exit_reason"].fillna("").str.contains("Time").sum()) if "exit_reason" in trades else 0,
    }
