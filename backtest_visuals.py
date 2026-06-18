from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def equity_curve_dataframe(trades: pd.DataFrame, initial_equity: float = 100000.0) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "equity", "drawdown_pct"])
    out = trades.copy()
    out["date"] = pd.to_datetime(out["exit_date"], errors="coerce")
    out["equity"] = pd.to_numeric(out.get("equity_after"), errors="coerce")
    if out["equity"].isna().all():
        out["equity"] = initial_equity + pd.to_numeric(out["pnl_dollars"], errors="coerce").fillna(0).cumsum()
    out = out.sort_values("date")
    out["drawdown_pct"] = ((out["equity"] - out["equity"].cummax()) / out["equity"].cummax()) * 100
    return out[["date", "equity", "drawdown_pct"]]


def equity_curve_chart(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["equity"], mode="lines", name="Equity", line=dict(color="#7dd3fc", width=2)))
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Equity")
    return fig


def drawdown_chart(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["drawdown_pct"], mode="lines", fill="tozeroy", name="Drawdown", line=dict(color="#fb7185", width=2)))
    fig.update_layout(template="plotly_dark", height=260, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Drawdown %")
    return fig
