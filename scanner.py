from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from ai_review import AIReviewSettings, review_dataframe_with_ai
from scan_logger import save_scan_results
from strategy_engine import STRATEGIES, best_strategy_per_ticker, build_strategy_results


def action_from_score(score: float | int | None) -> str:
    try:
        value = float(score)
    except Exception:
        return "Wait"
    if value >= 85:
        return "Strong watch"
    if value >= 75:
        return "Watch closely"
    if value >= 65:
        return "Conditional only"
    if value >= 50:
        return "Wait"
    return "Avoid for now"


def add_deterministic_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["ai_review_score"] = None
    out["final_watch_score"] = out["final_strategy_score"]
    out["ai_action"] = out["final_watch_score"].map(action_from_score)
    out["setup_quality"] = "Deterministic only"
    out["trade_maturity"] = "N/A"
    out["entry_quality"] = "N/A"
    out["stop_quality"] = "N/A"
    out["target_quality"] = "N/A"
    out["catalyst_interpretation"] = "N/A"
    out["confirmation_needed"] = ""
    out["main_reason"] = out.get("reasons", "")
    return out


def apply_scan_filters(
    df: pd.DataFrame,
    *,
    enabled_strategies: list[str] | None = None,
    category: str = "All",
    priority: str = "All",
    min_score: float = 60,
    min_price: float = 0,
    min_market_cap: float = 0,
    min_avg_volume: float = 0,
    min_rvol: float = 0,
    max_atr: float = 100,
    exclude_low_liquidity: bool = True,
    exclude_earnings_soon: bool = True,
    exclude_extended: bool = True,
    exclude_rejected_breakouts: bool = True,
    exclude_poor_rr: bool = False,
    require_above_200sma: bool = False,
    require_accepted_breakout: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in [
        "current_price",
        "market_cap",
        "average_volume_20d",
        "relative_volume",
        "atr_pct",
        "risk_flags",
        "sma_200",
        "breakout_acceptance_status",
    ]:
        if column not in out.columns:
            out[column] = pd.NA
    if enabled_strategies:
        out = out[out["strategy"].isin(enabled_strategies)]
    out = out[out["final_strategy_score"].fillna(0) >= min_score]
    if category != "All" and "category" in out:
        out = out[out["category"] == category]
    if priority != "All" and "priority" in out:
        out = out[out["priority"] == priority]
    out = out[(out["current_price"].isna()) | (out["current_price"] >= min_price)]
    out = out[(out["market_cap"].isna()) | (out["market_cap"] >= min_market_cap)]
    out = out[(out["average_volume_20d"].isna()) | (out["average_volume_20d"] >= min_avg_volume)]
    out = out[(out["relative_volume"].isna()) | (out["relative_volume"] >= min_rvol)]
    out = out[(out["atr_pct"].isna()) | (out["atr_pct"] <= max_atr)]
    if exclude_low_liquidity:
        out = out[~out["risk_flags"].fillna("").str.contains("Low Liquidity", case=False)]
    if exclude_extended:
        out = out[~out["risk_flags"].fillna("").str.contains("Extended", case=False)]
    if exclude_rejected_breakouts:
        out = out[~out["risk_flags"].fillna("").str.contains("Breakout Rejected|Liquidity Sweep Above High", regex=True)]
    if exclude_poor_rr:
        out = out[~out["risk_flags"].fillna("").str.contains("Poor Risk/Reward", case=False)]
    if exclude_earnings_soon and "next_earnings_date" in out:
        earnings = pd.to_datetime(out["next_earnings_date"], errors="coerce")
        today = datetime.now().date()
        cutoff = today + timedelta(days=2)
        out = out[~((earnings.dt.date >= today) & (earnings.dt.date <= cutoff))]
    if require_above_200sma:
        out = out[(out["current_price"].isna()) | (out["sma_200"].isna()) | (out["current_price"] >= out["sma_200"])]
    if require_accepted_breakout:
        out = out[out["breakout_acceptance_status"] == "accepted"]
    return out.sort_values("final_strategy_score", ascending=False).reset_index(drop=True)


def summarize_scan(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "unique_tickers": 0,
            "strong_watch": 0,
            "watch_or_better": 0,
            "best_ticker": "",
            "best_score": None,
            "market_regime": "N/A",
        }
    score_col = "final_watch_score" if "final_watch_score" in df else "final_strategy_score"
    top = df.sort_values(score_col, ascending=False).iloc[0]
    regimes = df.get("market_regime", pd.Series(dtype=str)).dropna()
    return {
        "rows": int(len(df)),
        "unique_tickers": int(df["ticker"].nunique()) if "ticker" in df else 0,
        "strong_watch": int((df[score_col].fillna(0) >= 85).sum()),
        "watch_or_better": int((df[score_col].fillna(0) >= 75).sum()),
        "best_ticker": top.get("ticker", ""),
        "best_score": top.get(score_col),
        "market_regime": regimes.mode().iloc[0] if not regimes.empty else "N/A",
    }


def run_full_watchlist_scan(
    dashboard_df: pd.DataFrame,
    *,
    enabled_strategies: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    ai_settings: AIReviewSettings | None = None,
    save_results: bool = True,
    best_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], str | None]:
    if dashboard_df.empty:
        return pd.DataFrame(), summarize_scan(pd.DataFrame()), None
    active = dashboard_df.copy()
    if "active" in active.columns:
        active = active[active["active"] == True].copy()
    strategy_df = build_strategy_results(active)
    filtered = apply_scan_filters(
        strategy_df,
        enabled_strategies=enabled_strategies or sorted(STRATEGIES.keys()),
        **(filters or {}),
    )
    if best_only:
        filtered = best_strategy_per_ticker(filtered)
    if ai_settings and ai_settings.enabled:
        reviewed = review_dataframe_with_ai(filtered.sort_values("final_strategy_score", ascending=False), ai_settings)
    else:
        reviewed = add_deterministic_review_columns(filtered)
    score_col = "final_watch_score" if "final_watch_score" in reviewed else "final_strategy_score"
    reviewed = reviewed.sort_values(score_col, ascending=False).reset_index(drop=True)
    summary = summarize_scan(reviewed)
    path = save_scan_results(reviewed, summary, "full_watchlist_scan") if save_results else None
    return reviewed, summary, str(path) if path else None
