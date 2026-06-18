from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from charting import create_trading_chart, get_cached_tickers
from data_yfinance import clear_watchlist_market_cache, get_multiple_price_data, get_price_data, get_watchlist_market_data
from indicators import add_indicators
from ai_review import AIReviewSettings, review_dataframe_with_ai
from ai_watchlist_curator import curate_weekly_watchlist
from backtest_ai_review import review_backtest_with_ai
from backtest_visuals import drawdown_chart, equity_curve_chart, equity_curve_dataframe
from backtester import BacktestConfig, run_backtest, save_backtest_results
from journal import (
    EXIT_REASON_OPTIONS,
    MISTAKE_TAG_OPTIONS,
    PLAN_STATUS_OPTIONS,
    SETUP_TYPE_OPTIONS,
    STRATEGY_OPTIONS,
    TRADE_GRADE_OPTIONS,
    actualize_trade,
    append_planned_trade,
    calculate_plan_fields,
    close_active_trade,
    create_planned_trade,
    delete_planned_trade,
    ensure_journal_files,
    grouped_performance,
    journal_coach_summary,
    load_planned_trades,
    load_trades,
    planned_vs_actual_table,
    plan_warnings,
    pnl_summary,
    run_ai_plan_review,
    save_uploaded_screenshot,
    update_planned_trade,
    validate_plan,
)
from ollama_client import list_ollama_models, summarize_setup
from ranking_model import rank_stocks
from scan_logger import list_scan_files, load_latest_scan, load_scan
from scanner import run_full_watchlist_scan
from strategy_presets import load_presets, save_preset
from strategy_engine import STRATEGIES, STRATEGY_DESCRIPTIONS, best_strategy_per_ticker, build_strategy_results
from watchlist_manager import (
    WATCHLIST_COLUMNS,
    add_category,
    add_watchlist_row,
    load_categories,
    load_watchlist,
    save_categories,
    save_watchlist,
)

st.set_page_config(page_title="Swing Trading Model", layout="wide")
st.title("Swing Trading Model")

PAGES = [
    "Single Stock Analysis",
    "Watchlist Ranking",
    "Watchlist Dashboard",
    "Strategy Scanner",
    "Comprehensive Scanner",
    "Weekly Trade Watchlist",
    "Backtesting Lab",
    "Trade Journal",
    "Chart Viewer",
]

if "page" not in st.session_state:
    st.session_state["page"] = "Single Stock Analysis"


def _page_changed() -> None:
    st.session_state["page"] = st.session_state["page_picker"]


page = st.sidebar.selectbox(
    "Choose page",
    PAGES,
    index=PAGES.index(st.session_state.get("page", "Single Stock Analysis")),
    key="page_picker",
    on_change=_page_changed,
)
page = st.session_state["page"]

use_cache = st.sidebar.checkbox("Use cached yfinance data", value=True)
st.sidebar.subheader("Ollama AI")
enable_ai = st.sidebar.checkbox("Enable Ollama AI Review", value=False)
allow_ai_adjustment = st.sidebar.checkbox("Allow AI score influence", value=True)
ai_weight = st.sidebar.slider("AI score weight", 0.0, 0.30, 0.15, 0.05)
allow_ai_caps = st.sidebar.checkbox("Allow AI caps/downgrades", value=True)
allow_positive_ai = st.sidebar.checkbox("Allow AI minor positive adjustment", value=True)
ai_candidate_limit = st.sidebar.number_input("Candidates to send to AI", min_value=1, max_value=50, value=25, step=1)
ai_min_score = st.sidebar.slider("Min deterministic score for AI review", 0, 100, 60)
ai_timeout = st.sidebar.number_input("Ollama timeout seconds", min_value=5, max_value=120, value=30, step=5)
force_ai_rerun = st.sidebar.checkbox("Force rerun AI review", value=False)
show_raw_ai_json = st.sidebar.checkbox("Show raw AI JSON", value=False)
ollama_url = st.sidebar.text_input("Ollama URL", "http://localhost:11434")
available_models = list_ollama_models(ollama_url, timeout=2) if enable_ai else []
if available_models:
    ollama_model = st.sidebar.selectbox("Ollama model", available_models)
else:
    ollama_model = st.sidebar.text_input("Ollama model", "llama3.2:latest")

ai_settings = AIReviewSettings(
    enabled=enable_ai,
    base_url=ollama_url,
    model=ollama_model,
    timeout=int(ai_timeout),
    candidate_limit=int(ai_candidate_limit),
    min_deterministic_score=float(ai_min_score),
    allow_score_influence=allow_ai_adjustment,
    ai_weight=float(ai_weight),
    allow_caps=allow_ai_caps,
    allow_positive_adjustment=allow_positive_ai,
    show_raw_json=show_raw_ai_json,
    force_rerun=force_ai_rerun,
)


@st.cache_data(ttl=900, show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    watchlist = load_watchlist()
    active = watchlist[watchlist["active"]]
    market_data = get_watchlist_market_data(active["ticker"].dropna().unique().tolist(), use_persistent_cache=use_cache)
    dashboard = watchlist.merge(market_data, on="ticker", how="left")
    dashboard["data_status"] = dashboard["data_status"].fillna("Data unavailable")
    dashboard["risk_flags"] = dashboard["risk_flags"].fillna("")
    manual_catalyst = dashboard["manual_catalyst"].fillna("").astype(str).str.strip() != ""
    needs_manual_flag = manual_catalyst & ~dashboard["risk_flags"].str.contains("Manual Catalyst", na=False)
    dashboard.loc[needs_manual_flag, "risk_flags"] = dashboard.loc[needs_manual_flag, "risk_flags"].apply(
        lambda value: "Manual Catalyst" if not str(value).strip() else f"{value}, Manual Catalyst"
    )
    dashboard["last_refreshed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    strategy_df = build_strategy_results(dashboard[dashboard["active"]])
    best = best_strategy_per_ticker(strategy_df)
    if not best.empty:
        best = best[
            [
                "ticker",
                "best_strategy",
                "final_strategy_score",
                "raw_strategy_score",
                "advanced_technical_score",
                "confluence_score",
                "context_score",
                "match_label",
                "breakout_acceptance_status",
                "liquidity_sweep_status",
                "fvg_lvn_status",
                "ob_hvn_status",
                "vwap_status",
                "volume_profile_location",
                "key_level_status",
                "trade_plan_type",
                "entry_zone",
                "stop_price",
                "target_1",
                "target_2",
                "risk_reward_target_1",
            ]
        ].rename(columns={"match_label": "best_strategy_label"})
        dashboard = dashboard.merge(best, on="ticker", how="left")
    return dashboard


def format_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    rename = {
        "ticker": "Ticker",
        "category": "Category",
        "quick_thesis": "Quick thesis",
        "macro_tag": "Macro tag",
        "manual_catalyst": "Manual catalyst",
        "priority": "Priority",
        "active": "Active",
        "current_price": "Current price",
        "return_1d_pct": "1D % change",
        "return_5d_pct": "5D % change",
        "return_1m_pct": "1M % change",
        "volume": "Volume",
        "relative_volume": "Relative volume",
        "market_cap": "Market cap",
        "market_cap_category": "Market cap category",
        "pe_ratio": "P/E",
        "forward_pe": "Forward P/E",
        "beta": "Beta",
        "atr_pct": "ATR %",
        "volatility_rating": "Volatility rating",
        "ema_8": "8 EMA",
        "ema_21": "21 EMA",
        "sma_50": "50 SMA",
        "sma_200": "200 SMA",
        "trend_status": "Trend status",
        "setup_type": "Setup type",
        "distance_to_52w_high_pct": "Distance to 52-week high",
        "distance_to_52w_low_pct": "Distance to 52-week low",
        "distance_to_nearest_support_pct": "Distance to nearest support",
        "distance_to_nearest_resistance_pct": "Distance to nearest resistance",
        "analyst_target_upside_pct": "Analyst target upside",
        "recommendation": "Recommendation",
        "next_earnings_date": "Next earnings date",
        "latest_headline": "Latest headline",
        "risk_flags": "Risk flags",
        "trade_readiness_score": "Trade readiness score",
        "trade_readiness_label": "Readiness label",
        "best_strategy": "Best strategy",
        "final_strategy_score": "Best strategy score",
        "advanced_technical_score": "Advanced technical score",
        "confluence_score": "Confluence score",
        "best_strategy_label": "Best strategy label",
        "breakout_acceptance_status": "Breakout acceptance",
        "liquidity_sweep_status": "Liquidity sweep",
        "fvg_lvn_status": "FVG confluence",
        "ob_hvn_status": "OB/HVN confluence",
        "vwap_status": "VWAP status",
        "volume_profile_location": "Volume profile location",
        "key_level_status": "Key level status",
        "trade_plan_type": "Trade plan type",
        "entry_zone": "Entry zone",
        "stop_price": "Stop",
        "target_1": "Target 1",
        "risk_reward_target_1": "Risk/reward",
        "data_status": "Data status",
    }
    columns = [column for column in rename if column in display.columns]
    display = display[columns].rename(columns=rename)
    numeric_columns = display.select_dtypes(include="number").columns
    for column in numeric_columns:
        display[column] = display[column].round(2)
    return display.fillna("N/A")


def journal_context_for_ticker(ticker: str) -> dict:
    if not ticker:
        return {}
    try:
        dashboard = load_dashboard_data()
        matches = dashboard[dashboard["ticker"].astype(str).str.upper() == ticker.upper()]
        if matches.empty:
            return {}
        row = matches.iloc[0]
        keys = [
            "current_price",
            "ema_8",
            "ema_9",
            "ema_21",
            "sma_200",
            "nearest_support",
            "nearest_resistance",
            "support_level",
            "breakout_level",
            "distance_from_8ema_pct",
            "distance_from_21ema_pct",
            "distance_from_200sma_pct",
            "distance_to_nearest_resistance_pct",
            "relative_volume",
            "atr_pct",
            "rsi_14",
            "market_regime",
            "best_strategy",
            "final_strategy_score",
            "final_watch_score",
            "risk_flags",
        ]
        return {key: row.get(key) for key in keys if key in row.index}
    except Exception:
        return {}


def analyze_ticker_control(df: pd.DataFrame, key_prefix: str) -> None:
    tickers = sorted(df["ticker"].dropna().astype(str).unique().tolist())
    if not tickers:
        return
    cols = st.columns([2, 1])
    selected = cols[0].selectbox("Ticker to analyze", tickers, key=f"{key_prefix}_ticker")
    if cols[1].button("Analyze Chart", key=f"{key_prefix}_analyze"):
        st.session_state["selected_ticker"] = selected
        selected_rows = df[df["ticker"] == selected]
        if not selected_rows.empty:
            selected_row = selected_rows.iloc[0]
            st.session_state["selected_trade_plan"] = {
                "entry_price_low": selected_row.get("entry_price_low"),
                "entry_price_high": selected_row.get("entry_price_high"),
                "stop_price": selected_row.get("stop_price"),
                "target_1": selected_row.get("target_1"),
                "target_2": selected_row.get("target_2"),
                "target_3": selected_row.get("target_3"),
                "vp_poc": selected_row.get("vp_poc"),
                "vp_value_area_high": selected_row.get("vp_value_area_high"),
                "vp_value_area_low": selected_row.get("vp_value_area_low"),
                "vp_nearest_hvn_above": selected_row.get("vp_nearest_hvn_above"),
                "bullish_fvg_low": selected_row.get("bullish_fvg_low"),
                "bullish_fvg_high": selected_row.get("bullish_fvg_high"),
                "bullish_order_block_low": selected_row.get("bullish_order_block_low"),
                "bullish_order_block_high": selected_row.get("bullish_order_block_high"),
                "intraday_vwap": selected_row.get("intraday_vwap"),
                "swept_level": selected_row.get("swept_level"),
            }
            st.session_state["selected_strategy_result"] = {
                "strategy": selected_row.get("strategy") or selected_row.get("best_strategy"),
                "score": selected_row.get("final_strategy_score"),
                "label": selected_row.get("match_label") or selected_row.get("best_strategy_label"),
                "reasons": selected_row.get("reasons"),
                "warnings": selected_row.get("warnings"),
                "final_watch_score": selected_row.get("final_watch_score"),
                "ai_action": selected_row.get("ai_action"),
                "setup_quality": selected_row.get("setup_quality"),
                "entry_comment": selected_row.get("entry_comment"),
                "stop_comment": selected_row.get("stop_comment"),
                "target_comment": selected_row.get("target_comment"),
                "confirmation_needed": selected_row.get("confirmation_needed"),
                "main_risks": selected_row.get("main_risks"),
                "raw_ai_json": selected_row.get("raw_ai_json"),
            }
        st.session_state["page"] = "Chart Viewer"
        st.rerun()


def maybe_add_ai_summary(df: pd.DataFrame) -> pd.DataFrame:
    if not enable_ai or df.empty:
        return df
    rows = []
    for _, row in df.head(20).iterrows():
        payload = {
            "ticker": row.get("ticker"),
            "category": row.get("category"),
            "manual_thesis": row.get("quick_thesis"),
            "manual_catalyst": row.get("manual_catalyst"),
            "strategy": row.get("strategy"),
            "deterministic_score": row.get("final_strategy_score"),
            "technical_reasons": row.get("reasons"),
            "warnings": row.get("warnings"),
            "risk_flags": row.get("risk_flags"),
            "advanced_technical_score": row.get("advanced_technical_score"),
            "confluence_score": row.get("confluence_score"),
            "liquidity_sweep_status": row.get("liquidity_sweep_status"),
            "swept_level": row.get("swept_level"),
            "breakout_acceptance_status": row.get("breakout_acceptance_status"),
            "fvg_lvn_status": row.get("fvg_lvn_status"),
            "ob_hvn_status": row.get("ob_hvn_status"),
            "volume_profile": {
                "poc": row.get("vp_poc"),
                "value_area_high": row.get("vp_value_area_high"),
                "value_area_low": row.get("vp_value_area_low"),
                "nearest_hvn_above": row.get("vp_nearest_hvn_above"),
                "nearest_lvn_above": row.get("vp_nearest_lvn_above"),
            },
            "vwap_status": row.get("vwap_status"),
            "estimated_fvg": row.get("bullish_fvg_midpoint"),
            "estimated_order_block": row.get("bullish_order_block_midpoint"),
            "entry_zone": row.get("entry_zone"),
            "stop": row.get("stop_price"),
            "target_1": row.get("target_1"),
            "target_2": row.get("target_2"),
            "latest_headline": row.get("latest_headline"),
            "earnings_date": row.get("next_earnings_date"),
            "data_quality_score": row.get("data_quality_score"),
        }
        ai = summarize_setup(payload, base_url=ollama_url, model=ollama_model, allow_adjustment=allow_ai_adjustment)
        combined = row.to_dict()
        adjustment = ai.get("ai_quality_adjustment", 0)
        combined["ai_adjustment"] = adjustment
        combined["ai_adjusted_score"] = max(0, min(100, (row.get("final_strategy_score") or 0) + adjustment))
        combined["ai_summary"] = ai.get("ai_summary")
        combined["ai_action"] = ai.get("watchlist_action")
        combined["ai_confidence"] = ai.get("confidence")
        rows.append(combined)
    return pd.DataFrame(rows)


def render_strategy_definitions() -> None:
    st.subheader("Strategy Definitions")
    for name, description in STRATEGY_DESCRIPTIONS.items():
        with st.expander(name):
            st.write(description)
    with st.expander("Higher-Timeframe Context Filter"):
        st.write(
            "This is not a standalone trade setup. It modifies every strategy using daily trend, SPY/QQQ regime, "
            "room to resistance, nearby support, ATR, and reward/risk. Very poor context caps final strategy scores."
        )


def apply_dashboard_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    filtered = df.copy()
    categories = ["All"] + sorted(filtered["category"].dropna().unique().tolist())
    priorities = ["All"] + sorted(filtered["priority"].dropna().unique().tolist())
    cols = st.columns(4)
    category = cols[0].selectbox("Category", categories, key=f"{key_prefix}_category")
    priority = cols[1].selectbox("Priority", priorities, key=f"{key_prefix}_priority")
    active_only = cols[2].checkbox("Active only", value=True, key=f"{key_prefix}_active")
    data_only = cols[3].checkbox("Hide data failures", value=False, key=f"{key_prefix}_data")
    if category != "All":
        filtered = filtered[filtered["category"] == category]
    if priority != "All":
        filtered = filtered[filtered["priority"] == priority]
    if active_only:
        filtered = filtered[filtered["active"]]
    if data_only:
        filtered = filtered[filtered["data_status"] == "OK"]
    return filtered


def render_watchlist_editor() -> None:
    st.subheader("Watchlist Editor")
    categories = load_categories()
    watchlist = load_watchlist()

    with st.expander("Add ticker", expanded=False):
        with st.form("add_ticker_form"):
            ticker = st.text_input("Ticker").upper()
            category = st.selectbox("Category", sorted(categories["category"].tolist()))
            quick_thesis = st.text_input("Quick thesis")
            macro_tag = st.text_input("Macro tag")
            manual_catalyst = st.text_input("Manual catalyst")
            priority = st.selectbox("Priority", ["High", "Medium", "Low"], index=1)
            submitted = st.form_submit_button("Add ticker")
            if submitted and ticker:
                add_watchlist_row(ticker, category, quick_thesis, macro_tag, manual_catalyst, priority, True)
                st.cache_data.clear()
                st.success(f"Added {ticker} to {category}.")
                st.rerun()

    with st.expander("Add category", expanded=False):
        with st.form("add_category_form"):
            new_category = st.text_input("New category")
            description = st.text_input("Description")
            submitted = st.form_submit_button("Add category")
            if submitted and new_category:
                add_category(new_category, description)
                st.success(f"Added category: {new_category}.")
                st.rerun()

    edited = st.data_editor(
        watchlist,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "active": st.column_config.CheckboxColumn("active"),
            "priority": st.column_config.SelectboxColumn("priority", options=["High", "Medium", "Low", ""]),
            "category": st.column_config.SelectboxColumn("category", options=sorted(categories["category"].tolist())),
        },
        key="watchlist_editor",
    )
    if st.button("Save watchlist changes"):
        save_watchlist(edited[WATCHLIST_COLUMNS])
        st.cache_data.clear()
        st.success("Watchlist saved.")
        st.rerun()

    with st.expander("Edit categories CSV", expanded=False):
        edited_categories = st.data_editor(categories, num_rows="dynamic", use_container_width=True, hide_index=True)
        if st.button("Save category changes"):
            save_categories(edited_categories)
            st.success("Categories saved.")
            st.rerun()


def render_category_summary(df: pd.DataFrame) -> None:
    active = df[df["active"]].copy()
    if active.empty:
        return
    rows = []
    for category, group in active.groupby("category"):
        ok = group[group["data_status"] == "OK"]
        rows.append(
            {
                "Category": category,
                "Active tickers": int(len(group)),
                "Best 1D mover": ok.sort_values("return_1d_pct", ascending=False)["ticker"].iloc[0] if not ok.empty else "N/A",
                "Best 5D mover": ok.sort_values("return_5d_pct", ascending=False)["ticker"].iloc[0] if not ok.empty else "N/A",
                "Highest RVOL": ok.sort_values("relative_volume", ascending=False)["ticker"].iloc[0] if not ok.empty else "N/A",
                "High-priority matches": int((group.get("trade_readiness_score", pd.Series(dtype=float)) >= 80).sum()),
                "Data failures": int((group["data_status"] != "OK").sum()),
                "Avg 1D %": round(float(ok["return_1d_pct"].mean()), 2) if not ok.empty else "N/A",
                "Avg 5D %": round(float(ok["return_5d_pct"].mean()), 2) if not ok.empty else "N/A",
            }
        )
    st.subheader("Category Summary")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if page == "Single Stock Analysis":
    ticker = st.text_input("Enter stock ticker", "AAPL").upper()

    if st.button("Analyze"):
        try:
            data = get_price_data(ticker, period="6mo", interval="1d", use_cache=use_cache)
            data = add_indicators(data)
            complete_rows = data.dropna()

            if complete_rows.empty:
                raise ValueError("Not enough price history to calculate indicators")

            latest = complete_rows.iloc[-1]

            st.subheader(f"{ticker} Summary")
            st.write(f"Latest close: ${latest['close']:.2f}")
            st.write(f"RSI 14: {latest['RSI_14']:.2f}")
            st.write(f"5-day return: {latest['return_5d'] * 100:.2f}%")
            st.write(f"20-day return: {latest['return_20d'] * 100:.2f}%")
            st.write(f"Volume ratio: {latest['volume_ratio']:.2f}")

            st.subheader("Recent Data")
            st.dataframe(data.tail())

            st.subheader("Close Price")
            st.line_chart(data["close"])

            st.subheader("Moving Averages")
            st.line_chart(data[["close", "SMA_20", "SMA_50", "EMA_20"]])

            st.subheader("RSI")
            st.line_chart(data["RSI_14"])

            st.subheader("Volume")
            st.bar_chart(data["volume"])

        except Exception as error:
            st.error(f"Could not analyze {ticker}: {error}")


elif page == "Watchlist Ranking":
    tickers_text = st.text_area(
        "Enter tickers separated by commas",
        "AAPL, MSFT, NVDA, AMD, META, GOOGL, AMZN, TSLA",
    )

    tickers = [ticker.strip().upper() for ticker in tickers_text.split(",") if ticker.strip()]

    if st.button("Rank Watchlist"):
        try:
            with st.spinner("Downloading/loading price data..."):
                price_data = get_multiple_price_data(tickers, period="6mo", interval="1d", use_cache=use_cache)

            ranking = rank_stocks(price_data)

            if ranking.empty:
                raise ValueError("No watchlist data could be loaded")

            st.subheader("Ranking Results")
            st.dataframe(ranking, use_container_width=True, hide_index=True)

            st.subheader("Top Candidates")
            st.dataframe(ranking.head(5), use_container_width=True, hide_index=True)

        except Exception as error:
            st.error(f"Could not rank watchlist: {error}")

elif page == "Watchlist Dashboard":
    st.subheader("Watchlist Dashboard")
    cols = st.columns([1, 2])
    if cols[0].button("Refresh yfinance Data"):
        clear_watchlist_market_cache()
        st.cache_data.clear()
        st.session_state["last_manual_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()

    with st.spinner("Loading watchlist market data..."):
        dashboard = load_dashboard_data()

    last_refreshed = st.session_state.get("last_manual_refresh") or dashboard["last_refreshed"].iloc[0]
    cols[1].caption(f"Last refreshed: {last_refreshed}")

    failures = dashboard[dashboard["data_status"] != "OK"]
    if not failures.empty:
        st.warning(f"{len(failures)} ticker/category rows have unavailable market data. The dashboard is still usable.")

    render_category_summary(dashboard)
    filtered = apply_dashboard_filters(dashboard, "dashboard")
    analyze_ticker_control(filtered, "dashboard")
    st.dataframe(format_dashboard(filtered), use_container_width=True, hide_index=True)

    render_watchlist_editor()

elif page == "Strategy Scanner":
    st.subheader("Strategy Scanner")
    render_strategy_definitions()

    with st.expander("Scanner Settings", expanded=True):
        cols = st.columns(6)
        enabled = cols[0].multiselect("Enabled strategies", sorted(STRATEGIES.keys()), default=sorted(STRATEGIES.keys()))
        min_price = cols[1].number_input("Min price", min_value=0.0, value=5.0, step=1.0)
        min_market_cap = cols[2].number_input("Min market cap $B", min_value=0.0, value=1.0, step=0.5)
        min_avg_volume = cols[3].number_input("Min avg volume", min_value=0, value=500000, step=100000)
        min_rvol = cols[4].number_input("Min RVOL", min_value=0.0, value=0.8, step=0.1)
        min_score = cols[5].slider("Score threshold", 0, 100, 50)
        max_atr = st.slider("Max ATR % filter", 0.0, 20.0, 12.0, 0.5)

    if st.button("Run Strategy Scan"):
        st.cache_data.clear()

    with st.spinner("Scanning active watchlist tickers..."):
        dashboard = load_dashboard_data()
        strategy_df = build_strategy_results(dashboard[dashboard["active"]])

    if enabled:
        strategy_df = strategy_df[strategy_df["strategy"].isin(enabled)]
    filtered = strategy_df[strategy_df["final_strategy_score"] >= min_score].copy()
    filtered = filtered[(filtered["current_price"].isna()) | (filtered["current_price"] >= min_price)]
    filtered = filtered[(filtered["market_cap"].isna()) | (filtered["market_cap"] >= min_market_cap * 1_000_000_000)]
    filtered = filtered[(filtered["average_volume_20d"].isna()) | (filtered["average_volume_20d"] >= min_avg_volume)]
    filtered = filtered[(filtered["relative_volume"].isna()) | (filtered["relative_volume"] >= min_rvol)]
    filtered = filtered[(filtered["atr_pct"].isna()) | (filtered["atr_pct"] <= max_atr)]

    analyze_ticker_control(filtered, "scanner")
    scanner_columns = [
        "ticker",
        "category",
        "strategy",
        "final_strategy_score",
        "raw_strategy_score",
        "advanced_technical_score",
        "confluence_score",
        "context_score",
        "match_label",
        "watchlist_priority",
        "current_price",
        "relative_volume",
        "atr_pct",
        "rsi_14",
        "entry_zone",
        "stop_price",
        "target_1",
        "target_2",
        "risk_reward_target_1",
        "breakout_acceptance_status",
        "liquidity_sweep_status",
        "fvg_lvn_status",
        "ob_hvn_status",
        "vwap_status",
        "volume_profile_location",
        "risk_flags",
        "reasons",
        "warnings",
    ]
    visible = filtered[[column for column in scanner_columns if column in filtered.columns]].copy()
    for column in visible.select_dtypes(include="number").columns:
        visible[column] = visible[column].round(2)
    st.dataframe(visible.fillna("N/A"), use_container_width=True, hide_index=True)

    with st.expander("Per-ticker strategy details"):
        for ticker, group in filtered.groupby("ticker"):
            st.markdown(f"**{ticker}**")
            detail_columns = [
                "strategy",
                "final_strategy_score",
                "advanced_technical_score",
                "confluence_score",
                "match_label",
                "entry_zone",
                "entry_zone_source",
                "stop_price",
                "target_1",
                "breakout_acceptance_status",
                "liquidity_sweep_status",
                "fvg_lvn_status",
                "ob_hvn_status",
                "risk_flags",
                "reasons",
                "warnings",
            ]
            st.dataframe(group[[column for column in detail_columns if column in group.columns]].fillna("N/A"), use_container_width=True, hide_index=True)

elif page == "Comprehensive Scanner":
    st.subheader("Comprehensive Scanner")
    render_strategy_definitions()

    watchlist = load_watchlist()
    active_watchlist = watchlist[watchlist["active"]].copy()
    category_options = ["All"] + sorted(active_watchlist["category"].dropna().astype(str).unique().tolist())
    priority_options = ["All"] + sorted(active_watchlist["priority"].dropna().astype(str).unique().tolist())

    with st.expander("Scan controls", expanded=True):
        cols = st.columns(4)
        scan_strategies = cols[0].multiselect("Strategies", sorted(STRATEGIES.keys()), default=sorted(STRATEGIES.keys()))
        scan_category = cols[1].selectbox("Category", category_options)
        scan_priority = cols[2].selectbox("Priority", priority_options)
        scan_best_only = cols[3].checkbox("Best strategy per ticker", value=True)

        filter_cols = st.columns(6)
        scan_min_score = filter_cols[0].slider("Min score", 0, 100, 60)
        scan_min_price = filter_cols[1].number_input("Min price", min_value=0.0, value=5.0, step=1.0)
        scan_min_market_cap_b = filter_cols[2].number_input("Min market cap $B", min_value=0.0, value=1.0, step=0.5)
        scan_min_volume = filter_cols[3].number_input("Min avg volume", min_value=0, value=500000, step=100000)
        scan_min_rvol = filter_cols[4].number_input("Min RVOL", min_value=0.0, value=0.8, step=0.1)
        scan_max_atr = filter_cols[5].number_input("Max ATR %", min_value=0.0, value=12.0, step=0.5)

        rule_cols = st.columns(6)
        scan_exclude_low_liquidity = rule_cols[0].checkbox("Exclude low liquidity", value=True)
        scan_exclude_earnings = rule_cols[1].checkbox("Exclude earnings soon", value=True)
        scan_exclude_extended = rule_cols[2].checkbox("Exclude extended", value=True)
        scan_exclude_rejected = rule_cols[3].checkbox("Exclude rejected breakouts", value=True)
        scan_exclude_poor_rr = rule_cols[4].checkbox("Exclude poor R/R", value=False)
        scan_require_above_200 = rule_cols[5].checkbox("Require above 200 SMA", value=False)
        scan_require_accepted = st.checkbox("Require accepted breakout", value=False)

    run_scan = st.button("Run Full Watchlist Scan", type="primary")
    if st.button("Refresh yfinance Data", key="comprehensive_refresh"):
        clear_watchlist_market_cache()
        st.cache_data.clear()
        st.session_state.pop("last_full_scan", None)
        st.rerun()

    scan_filters = {
        "category": scan_category,
        "priority": scan_priority,
        "min_score": scan_min_score,
        "min_price": scan_min_price,
        "min_market_cap": scan_min_market_cap_b * 1_000_000_000,
        "min_avg_volume": scan_min_volume,
        "min_rvol": scan_min_rvol,
        "max_atr": scan_max_atr,
        "exclude_low_liquidity": scan_exclude_low_liquidity,
        "exclude_earnings_soon": scan_exclude_earnings,
        "exclude_extended": scan_exclude_extended,
        "exclude_rejected_breakouts": scan_exclude_rejected,
        "exclude_poor_rr": scan_exclude_poor_rr,
        "require_above_200sma": scan_require_above_200,
        "require_accepted_breakout": scan_require_accepted,
    }

    if run_scan:
        with st.spinner("Running full watchlist scan..."):
            dashboard = load_dashboard_data()
            results, summary, saved_path = run_full_watchlist_scan(
                dashboard,
                enabled_strategies=scan_strategies,
                filters=scan_filters,
                ai_settings=ai_settings,
                save_results=True,
                best_only=scan_best_only,
            )
        st.session_state["last_full_scan"] = results
        st.session_state["last_full_scan_summary"] = summary
        st.session_state["last_full_scan_path"] = saved_path

    saved_scans = list_scan_files()
    if saved_scans:
        with st.expander("Saved scans", expanded=False):
            selected_scan = st.selectbox("Open saved scan", [path.name for path in saved_scans])
            if st.button("Load selected scan"):
                path = next(path for path in saved_scans if path.name == selected_scan)
                loaded = load_scan(path)
                st.session_state["last_full_scan"] = loaded
                st.session_state["last_full_scan_summary"] = {}
                st.session_state["last_full_scan_path"] = str(path)

    scan_results = st.session_state.get("last_full_scan", pd.DataFrame())
    scan_summary = st.session_state.get("last_full_scan_summary", {})
    if scan_results.empty:
        latest = load_latest_scan()
        if not latest.empty:
            st.info("Showing latest saved scan. Click Run Full Watchlist Scan for a fresh scan.")
            scan_results = latest

    if scan_results.empty:
        st.info("No full scan results yet. Run the scanner when you are ready.")
    else:
        score_col = "final_watch_score" if "final_watch_score" in scan_results else "final_strategy_score"
        metric_cols = st.columns(5)
        metric_cols[0].metric("Rows", scan_summary.get("rows", len(scan_results)))
        metric_cols[1].metric("Tickers", scan_summary.get("unique_tickers", scan_results["ticker"].nunique()))
        metric_cols[2].metric("Strong watch", scan_summary.get("strong_watch", int((scan_results[score_col].fillna(0) >= 85).sum())))
        metric_cols[3].metric("Watch+", scan_summary.get("watch_or_better", int((scan_results[score_col].fillna(0) >= 75).sum())))
        metric_cols[4].metric("Market", scan_summary.get("market_regime", "N/A"))
        if st.session_state.get("last_full_scan_path"):
            st.caption(f"Saved: {st.session_state['last_full_scan_path']}")

        if enable_ai and not scan_results.empty:
            with st.expander("AI Weekly Curator", expanded=True):
                curation = curate_weekly_watchlist(
                    scan_results.sort_values(score_col, ascending=False),
                    base_url=ollama_url,
                    model=ollama_model,
                    timeout=int(ai_timeout),
                    limit=min(25, int(ai_candidate_limit)),
                )
                st.write(curation.get("weekly_market_summary", ""))
                curator_cols = st.columns(5)
                for col, key, title in [
                    (curator_cols[0], "strong_watch", "Strong Watch"),
                    (curator_cols[1], "watch_closely", "Watch Closely"),
                    (curator_cols[2], "conditional_only", "Conditional Only"),
                    (curator_cols[3], "do_not_chase", "Do Not Chase"),
                    (curator_cols[4], "avoid_for_now", "Avoid For Now"),
                ]:
                    col.markdown(f"**{title}**")
                    for item in curation.get(key, [])[:8]:
                        col.caption(f"{item.get('ticker', '')}: {item.get('reason', '')}")

        analyze_ticker_control(scan_results, "comprehensive_scan")
        scan_columns = [
            "ticker",
            "category",
            "strategy",
            "final_watch_score",
            "final_strategy_score",
            "ai_review_score",
            "ai_action",
            "setup_quality",
            "current_price",
            "relative_volume",
            "atr_pct",
            "rsi_14",
            "entry_zone",
            "stop_price",
            "target_1",
            "target_2",
            "risk_reward_target_1",
            "breakout_acceptance_status",
            "liquidity_sweep_status",
            "fvg_lvn_status",
            "ob_hvn_status",
            "vwap_status",
            "risk_flags",
            "main_reason",
            "reasons",
            "warnings",
        ]
        visible = scan_results[[column for column in scan_columns if column in scan_results.columns]].copy()
        for column in visible.select_dtypes(include="number").columns:
            visible[column] = visible[column].round(2)
        st.dataframe(visible.fillna("N/A"), use_container_width=True, hide_index=True)
        st.download_button(
            "Download scan CSV",
            data=scan_results.to_csv(index=False),
            file_name=f"full_watchlist_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

elif page == "Weekly Trade Watchlist":
    st.subheader("Weekly Trade Watchlist")
    if st.button("Refresh yfinance Data", key="weekly_refresh"):
        clear_watchlist_market_cache()
        st.cache_data.clear()
        st.session_state["last_manual_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.rerun()

    with st.spinner("Scoring watchlist strategies..."):
        dashboard = load_dashboard_data()
        strategy_df = build_strategy_results(dashboard[dashboard["active"]])
        strategy_df = best_strategy_per_ticker(strategy_df)

    if strategy_df.empty:
        st.info("No active strategy rows are available yet.")
    else:
        cols = st.columns(7)
        strategy_options = ["All Strategies"] + sorted(STRATEGIES.keys())
        strategy = cols[0].selectbox("Strategy", strategy_options)
        minimum_score = cols[1].slider("Minimum score", 0, 100, 50)
        category = cols[2].selectbox("Category", ["All"] + sorted(strategy_df["category"].dropna().unique().tolist()))
        priority = cols[3].selectbox("Priority", ["All"] + sorted(strategy_df["priority"].dropna().unique().tolist()))
        exclude_low_liquidity = cols[4].checkbox("Exclude low liquidity", value=False)
        exclude_earnings = cols[5].checkbox("Exclude earnings within 2 days", value=True)
        exclude_extended = cols[6].checkbox("Exclude extended", value=False)
        advanced_cols = st.columns(6)
        require_accepted_breakout = advanced_cols[0].checkbox("Require accepted breakout", value=False)
        require_sweep = advanced_cols[1].checkbox("Require liquidity sweep/reclaim", value=False)
        require_confluence = advanced_cols[2].checkbox("Require FVG/OB confluence", value=False)
        require_vwap = advanced_cols[3].checkbox("Require above VWAP proxy", value=False)
        exclude_rejected_breakouts = advanced_cols[4].checkbox("Exclude rejected breakouts", value=True)
        exclude_poor_rr = advanced_cols[5].checkbox("Exclude poor R/R", value=False)

        filtered = strategy_df[strategy_df["final_strategy_score"] >= minimum_score].copy()
        if strategy != "All Strategies":
            filtered = filtered[filtered["strategy"] == strategy]
        if category != "All":
            filtered = filtered[filtered["category"] == category]
        if priority != "All":
            filtered = filtered[filtered["priority"] == priority]
        if exclude_earnings:
            earnings = pd.to_datetime(filtered["next_earnings_date"], errors="coerce")
            filtered = filtered[~((earnings.dt.date >= datetime.now().date()) & (earnings.dt.date <= (datetime.now() + timedelta(days=2)).date()))]
        if exclude_extended:
            filtered = filtered[~filtered["risk_flags"].fillna("").str.contains("Extended")]
        if exclude_low_liquidity:
            filtered = filtered[~filtered["risk_flags"].fillna("").str.contains("Low Liquidity")]
        if require_accepted_breakout:
            filtered = filtered[filtered["breakout_acceptance_status"] == "accepted"]
        if require_sweep:
            filtered = filtered[filtered["bullish_liquidity_sweep_detected"] == True]
        if require_confluence:
            filtered = filtered[
                filtered["fvg_lvn_status"].fillna("").str.contains("FVG")
                | filtered["ob_hvn_status"].fillna("").str.contains("OB")
                | (filtered["confluence_score"].fillna(0) >= 30)
            ]
        if require_vwap:
            filtered = filtered[filtered["price_above_vwap"] == True]
        if exclude_rejected_breakouts:
            filtered = filtered[~filtered["risk_flags"].fillna("").str.contains("Breakout Rejected|Liquidity Sweep Above High", regex=True)]
        if exclude_poor_rr:
            filtered = filtered[~filtered["risk_flags"].fillna("").str.contains("Poor Risk/Reward")]

        if enable_ai:
            with st.spinner("Running AI setup review on qualified candidates..."):
                filtered = review_dataframe_with_ai(filtered.sort_values("final_strategy_score", ascending=False), ai_settings)
            unavailable = filtered[filtered.get("ai_available", False) == False] if "ai_available" in filtered else pd.DataFrame()
            if not unavailable.empty:
                st.warning("Ollama unavailable or disabled for some candidates; using deterministic scan only where needed.")
        else:
            filtered["ai_review_score"] = None
            filtered["final_watch_score"] = filtered["final_strategy_score"]
            filtered["ai_action"] = "Deterministic only"
            filtered["setup_quality"] = "N/A"
            filtered["trade_maturity"] = "N/A"
            filtered["entry_quality"] = "N/A"
            filtered["stop_quality"] = "N/A"
            filtered["target_quality"] = "N/A"
            filtered["catalyst_interpretation"] = "N/A"
            filtered["confirmation_needed"] = ""
            filtered["main_reason"] = filtered["reasons"]

        if enable_ai and not filtered.empty:
            with st.expander("AI Weekly Curator", expanded=True):
                curation = curate_weekly_watchlist(
                    filtered.sort_values("final_watch_score", ascending=False),
                    base_url=ollama_url,
                    model=ollama_model,
                    timeout=int(ai_timeout),
                    limit=min(25, int(ai_candidate_limit)),
                )
                st.write(curation.get("weekly_market_summary", ""))
                curator_cols = st.columns(5)
                for col, key, title in [
                    (curator_cols[0], "strong_watch", "Strong Watch"),
                    (curator_cols[1], "watch_closely", "Watch Closely"),
                    (curator_cols[2], "conditional_only", "Conditional Only"),
                    (curator_cols[3], "do_not_chase", "Do Not Chase"),
                    (curator_cols[4], "avoid_for_now", "Avoid For Now"),
                ]:
                    col.markdown(f"**{title}**")
                    for item in curation.get(key, [])[:8]:
                        col.caption(f"{item.get('ticker', '')}: {item.get('reason', '')}")

        analyze_ticker_control(filtered, "weekly")
        columns = [
            "ticker",
            "category",
            "best_strategy",
            "strategy",
            "final_watch_score",
            "final_strategy_score",
            "ai_review_score",
            "ai_action",
            "setup_quality",
            "trade_maturity",
            "entry_quality",
            "stop_quality",
            "target_quality",
            "catalyst_interpretation",
            "confirmation_needed",
            "raw_strategy_score",
            "advanced_technical_score",
            "context_score",
            "match_label",
            "entry_type",
            "setup_type",
            "trend_status",
            "current_price",
            "return_1d_pct",
            "return_5d_pct",
            "relative_volume",
            "atr_pct",
            "rsi_14",
            "distance_from_8ema_pct",
            "distance_from_9ema_pct",
            "distance_from_21ema_pct",
            "distance_from_200sma_pct",
            "distance_to_nearest_resistance_pct",
            "distance_to_nearest_support_pct",
            "entry_zone",
            "stop_price",
            "target_1",
            "target_2",
            "risk_reward_target_1",
            "breakout_acceptance_status",
            "liquidity_sweep_status",
            "fvg_lvn_status",
            "ob_hvn_status",
            "vwap_status",
            "volume_profile_location",
            "entry_zone_source",
            "stop_source",
            "target_source",
            "risk_flags",
            "main_reason",
            "reasons",
            "warnings",
            "manual_catalyst",
            "latest_headline",
            "ai_adjustment",
            "ai_adjusted_score",
            "ai_summary",
            "ai_action",
            "raw_ai_json",
        ]
        visible = filtered[[column for column in columns if column in filtered.columns]].copy()
        for column in visible.select_dtypes(include="number").columns:
            visible[column] = visible[column].round(2)
        st.dataframe(visible.fillna("N/A"), use_container_width=True, hide_index=True)

elif page == "Backtesting Lab":
    st.subheader("Backtesting Lab")
    watchlist = load_watchlist()
    active_tickers = sorted(watchlist[watchlist["active"]]["ticker"].dropna().astype(str).str.upper().unique().tolist())
    presets = load_presets()
    selected_preset = st.selectbox("Strategy preset", ["Custom"] + sorted(presets.keys()))
    preset_config = presets.get(selected_preset, {}).get("config", {}) if selected_preset != "Custom" else {}

    with st.expander("Backtest setup", expanded=True):
        cols = st.columns(4)
        default_tickers = preset_config.get("tickers") or active_tickers[:5]
        bt_tickers = cols[0].multiselect("Universe", active_tickers, default=[ticker for ticker in default_tickers if ticker in active_tickers])
        bt_strategy = cols[1].selectbox(
            "Strategy",
            ["All Strategies"] + sorted(STRATEGIES.keys()),
            index=(["All Strategies"] + sorted(STRATEGIES.keys())).index(preset_config.get("strategy", "All Strategies"))
            if preset_config.get("strategy", "All Strategies") in ["All Strategies"] + sorted(STRATEGIES.keys())
            else 0,
        )
        bt_start = cols[2].date_input("Start date", value=pd.to_datetime(preset_config.get("start_date", "2024-01-01")).date())
        bt_end = cols[3].date_input("End date", value=pd.to_datetime(preset_config.get("end_date", datetime.now().date())).date())

        rule_cols = st.columns(4)
        bt_entry = rule_cols[0].selectbox("Entry rule", ["Next open", "Next close"], index=0)
        bt_stop = rule_cols[1].selectbox("Stop rule", ["Strategy stop", "3% fixed", "5% fixed"], index=0)
        bt_target = rule_cols[2].selectbox("Target rule", ["Target 1", "Target 2", "2R"], index=0)
        bt_hold = rule_cols[3].number_input("Max holding days", min_value=1, max_value=90, value=int(preset_config.get("max_holding_days", 20)), step=1)

        filter_cols = st.columns(5)
        bt_min_score = filter_cols[0].slider("Min signal score", 0, 100, int(preset_config.get("min_score", 65)))
        bt_min_rvol = filter_cols[1].number_input("Min RVOL", min_value=0.0, value=float(preset_config.get("min_rvol", 0.8)), step=0.1)
        bt_max_atr = filter_cols[2].number_input("Max ATR %", min_value=0.0, value=float(preset_config.get("max_atr_pct", 14.0)), step=0.5)
        bt_require_200 = filter_cols[3].checkbox("Require above 200 SMA", value=bool(preset_config.get("require_above_200sma", False)))
        bt_overlap = filter_cols[4].checkbox("Allow overlapping trades", value=bool(preset_config.get("allow_overlapping_trades", False)))

        risk_cols = st.columns(4)
        bt_initial = risk_cols[0].number_input("Initial equity", min_value=1000.0, value=float(preset_config.get("initial_equity", 100000.0)), step=5000.0)
        bt_risk = risk_cols[1].number_input("Risk per trade %", min_value=0.1, max_value=10.0, value=float(preset_config.get("risk_per_trade_pct", 1.0)), step=0.1)
        bt_slippage = risk_cols[2].number_input("Slippage %", min_value=0.0, max_value=2.0, value=float(preset_config.get("slippage_pct", 0.05)), step=0.01)
        bt_commission = risk_cols[3].number_input("Commission per trade", min_value=0.0, value=float(preset_config.get("commission_per_trade", 0.0)), step=0.5)

    bt_config_dict = {
        "tickers": bt_tickers,
        "strategy": bt_strategy,
        "start_date": str(bt_start),
        "end_date": str(bt_end),
        "min_score": bt_min_score,
        "min_rvol": bt_min_rvol,
        "max_atr_pct": bt_max_atr,
        "require_above_200sma": bt_require_200,
        "entry_rule": bt_entry,
        "stop_rule": bt_stop,
        "target_rule": bt_target,
        "max_holding_days": int(bt_hold),
        "initial_equity": bt_initial,
        "risk_per_trade_pct": bt_risk,
        "commission_per_trade": bt_commission,
        "slippage_pct": bt_slippage,
        "allow_overlapping_trades": bt_overlap,
        "use_cache": use_cache,
    }

    preset_cols = st.columns([2, 1])
    preset_name = preset_cols[0].text_input("Save current settings as preset", value="")
    if preset_cols[1].button("Save preset") and preset_name.strip():
        save_preset(preset_name, bt_config_dict)
        st.success(f"Saved preset: {preset_name}")

    run_bt = st.button("Run Backtest", type="primary")
    if run_bt:
        if not bt_tickers:
            st.warning("Choose at least one ticker.")
        else:
            config = BacktestConfig(**bt_config_dict)
            with st.spinner("Running historical strategy test..."):
                trades, metrics = run_backtest(config)
                saved_path = save_backtest_results(trades, metrics, config)
            st.session_state["last_backtest_trades"] = trades
            st.session_state["last_backtest_metrics"] = metrics
            st.session_state["last_backtest_path"] = str(saved_path)

    trades = st.session_state.get("last_backtest_trades", pd.DataFrame())
    metrics = st.session_state.get("last_backtest_metrics", {})
    if trades.empty:
        st.info("No backtest results yet. Pick a universe and run the test.")
    else:
        metric_cols = st.columns(6)
        metric_cols[0].metric("Trades", metrics.get("total_trades", 0))
        metric_cols[1].metric("Win rate", f"{metrics.get('win_rate', 0)}%")
        metric_cols[2].metric("Avg R", metrics.get("avg_r", 0))
        metric_cols[3].metric("Profit factor", metrics.get("profit_factor", 0))
        metric_cols[4].metric("Return", f"{metrics.get('total_return_pct', 0)}%")
        metric_cols[5].metric("Max DD", f"{metrics.get('max_drawdown_pct', 0)}%")
        if st.session_state.get("last_backtest_path"):
            st.caption(f"Saved: {st.session_state['last_backtest_path']}")

        curve = equity_curve_dataframe(trades, float(bt_initial))
        chart_cols = st.columns([2, 1])
        chart_cols[0].plotly_chart(equity_curve_chart(curve), use_container_width=True)
        chart_cols[1].plotly_chart(drawdown_chart(curve), use_container_width=True)

        display_trades = trades.copy()
        for column in display_trades.select_dtypes(include="number").columns:
            display_trades[column] = display_trades[column].round(2)
        st.dataframe(display_trades.fillna("N/A"), use_container_width=True, hide_index=True)
        st.download_button(
            "Download backtest trades CSV",
            data=trades.to_csv(index=False),
            file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        if enable_ai and st.button("Review Backtest With Ollama"):
            with st.spinner("Asking Ollama to review the backtest..."):
                ai_backtest = review_backtest_with_ai(metrics, trades, base_url=ollama_url, model=ollama_model, timeout=int(ai_timeout))
            st.json(ai_backtest)

    with st.expander("Parameter sweep", expanded=False):
        sweep_scores = st.text_input("Score thresholds", value="60,65,70,75")
        sweep_rvols = st.text_input("RVOL thresholds", value="0.8,1.0,1.2")
        if st.button("Run Parameter Sweep"):
            rows = []
            score_values = [float(item.strip()) for item in sweep_scores.split(",") if item.strip()]
            rvol_values = [float(item.strip()) for item in sweep_rvols.split(",") if item.strip()]
            for score_value in score_values:
                for rvol_value in rvol_values:
                    config = BacktestConfig(**{**bt_config_dict, "min_score": score_value, "min_rvol": rvol_value})
                    with st.spinner(f"Testing score {score_value}, RVOL {rvol_value}..."):
                        sweep_trades, sweep_metrics = run_backtest(config)
                    rows.append(
                        {
                            "min_score": score_value,
                            "min_rvol": rvol_value,
                            "trades": sweep_metrics.get("total_trades", 0),
                            "win_rate": sweep_metrics.get("win_rate", 0),
                            "avg_r": sweep_metrics.get("avg_r", 0),
                            "profit_factor": sweep_metrics.get("profit_factor", 0),
                            "return_pct": sweep_metrics.get("total_return_pct", 0),
                            "max_drawdown_pct": sweep_metrics.get("max_drawdown_pct", 0),
                        }
                    )
            sweep_df = pd.DataFrame(rows).sort_values(["return_pct", "avg_r"], ascending=False)
            st.dataframe(sweep_df, use_container_width=True, hide_index=True)

elif page == "Trade Journal":
    st.subheader("Trade Journal")
    ensure_journal_files()

    planned_df = load_planned_trades()
    trades_df = load_trades()

    tabs = st.tabs(
        [
            "Plan New Trade",
            "Planned Trades",
            "Active Trades",
            "Closed Trade Journal",
            "P&L Summary",
            "AI Journal Coach",
            "Personal Edge",
            "Planned vs Actual Review",
        ]
    )

    with tabs[0]:
        st.markdown("**Plan New Trade**")
        with st.form("plan_new_trade_form", clear_on_submit=False):
            cols = st.columns(4)
            setup_date = cols[0].date_input("Setup date", value=datetime.now().date())
            ticker = cols[1].text_input("Ticker").upper().strip()
            strategy_name = cols[2].selectbox("Strategy", STRATEGY_OPTIONS)
            setup_type = cols[3].selectbox("Setup type", SETUP_TYPE_OPTIONS)

            price_cols = st.columns(4)
            planned_entry_price = price_cols[0].number_input("Planned entry price", min_value=0.0, value=0.0, step=0.01)
            planned_stop_loss = price_cols[1].number_input("Planned stop loss", min_value=0.0, value=0.0, step=0.01)
            planned_exit_price = price_cols[2].number_input("Planned exit / target price", min_value=0.0, value=0.0, step=0.01)
            planned_shares = price_cols[3].number_input("Planned shares", min_value=0, value=0, step=1)

            preview_errors = validate_plan(
                ticker,
                setup_date,
                strategy_name,
                setup_type,
                float(planned_entry_price),
                float(planned_stop_loss),
                float(planned_exit_price),
                int(planned_shares),
            )
            if not preview_errors:
                preview = calculate_plan_fields(float(planned_entry_price), float(planned_stop_loss), float(planned_exit_price), int(planned_shares))
                metric_cols = st.columns(6)
                metric_cols[0].metric("Exposure", f"${preview['gross_exposure']:,.2f}")
                metric_cols[1].metric("Max Loss", f"${preview['max_loss_dollars']:,.2f}")
                metric_cols[2].metric("Max Gain", f"${preview['max_gain_dollars']:,.2f}")
                metric_cols[3].metric("R/R", preview["planned_risk_reward_ratio"])
                metric_cols[4].metric("Target %", f"{preview['planned_return_pct_to_target']}%")
                metric_cols[5].metric("Stop %", f"{preview['planned_loss_pct_to_stop']}%")
                for warning in plan_warnings(preview):
                    if "acceptable" in warning.lower():
                        st.success(warning)
                    else:
                        st.warning(warning)
                context = journal_context_for_ticker(ticker)
                nearest_resistance = pd.to_numeric(context.get("nearest_resistance"), errors="coerce")
                ema_distance = pd.to_numeric(context.get("distance_from_8ema_pct"), errors="coerce")
                if pd.notna(nearest_resistance) and nearest_resistance > 0:
                    resistance_gap = ((nearest_resistance - planned_entry_price) / planned_entry_price) * 100
                    if 0 <= resistance_gap <= 3:
                        st.warning("Planned entry is close to known resistance.")
                if pd.notna(ema_distance) and ema_distance > 8:
                    st.warning("Possible chase risk: price looks extended from the 8 EMA.")
            else:
                for error in preview_errors:
                    st.caption(error)

            submit_cols = st.columns(3)
            save_plan = submit_cols[0].form_submit_button("Save Planned Trade")
            save_and_review = submit_cols[1].form_submit_button("Save Planned Trade and Run AI Plan Review")
            clear_plan = submit_cols[2].form_submit_button("Clear Form")

        if clear_plan:
            st.rerun()

        if save_plan or save_and_review:
            errors = validate_plan(
                ticker,
                setup_date,
                strategy_name,
                setup_type,
                float(planned_entry_price),
                float(planned_stop_loss),
                float(planned_exit_price),
                int(planned_shares),
            )
            if errors:
                for error in errors:
                    st.error(error)
            else:
                record = create_planned_trade(
                    setup_date,
                    ticker,
                    strategy_name,
                    setup_type,
                    float(planned_entry_price),
                    float(planned_stop_loss),
                    float(planned_exit_price),
                    int(planned_shares),
                )
                if save_and_review:
                    with st.spinner("Running AI pre-trade review..."):
                        review_packet = {**record, "scanner_chart_context": journal_context_for_ticker(record["ticker"])}
                        record.update(run_ai_plan_review(review_packet, base_url=ollama_url, model=ollama_model, timeout=int(ai_timeout)))
                append_planned_trade(record)
                st.success(f"Saved planned trade {record['plan_id']}.")
                st.cache_data.clear()

    with tabs[1]:
        st.markdown("**Planned Trades**")
        planned_df = load_planned_trades()
        planned_only = planned_df[planned_df["plan_status"].isin(["Planned", "Cancelled", "Skipped"])].copy() if not planned_df.empty else pd.DataFrame()

        if planned_only.empty:
            st.info("No planned trades yet.")
        else:
            filter_cols = st.columns(6)
            status_filter = filter_cols[0].selectbox("Status", ["All"] + PLAN_STATUS_OPTIONS, key="planned_status_filter")
            ticker_filter = filter_cols[1].text_input("Ticker filter", key="planned_ticker_filter").upper().strip()
            strategy_filter = filter_cols[2].selectbox("Strategy", ["All"] + STRATEGY_OPTIONS, key="planned_strategy_filter")
            setup_filter = filter_cols[3].selectbox("Setup type", ["All"] + SETUP_TYPE_OPTIONS, key="planned_setup_filter")
            min_rr = filter_cols[4].number_input("Minimum R/R", min_value=0.0, value=0.0, step=0.1, key="planned_min_rr")
            confidence_filter = filter_cols[5].selectbox("AI confidence", ["All", "High", "Medium", "Low"], key="planned_conf_filter")

            filtered = planned_only.copy()
            if status_filter != "All":
                filtered = filtered[filtered["plan_status"] == status_filter]
            if ticker_filter:
                filtered = filtered[filtered["ticker"].astype(str).str.contains(ticker_filter, case=False, na=False)]
            if strategy_filter != "All":
                filtered = filtered[filtered["strategy_name"] == strategy_filter]
            if setup_filter != "All":
                filtered = filtered[filtered["setup_type"] == setup_filter]
            if min_rr:
                filtered = filtered[pd.to_numeric(filtered["planned_risk_reward_ratio"], errors="coerce").fillna(0) >= min_rr]
            if confidence_filter != "All":
                filtered = filtered[filtered["ai_plan_confidence"].astype(str) == confidence_filter]

            columns = {
                "setup_date": "Setup Date",
                "ticker": "Ticker",
                "plan_status": "Status",
                "strategy_name": "Strategy",
                "setup_type": "Setup Type",
                "planned_entry_price": "Planned Entry",
                "planned_stop_loss": "Planned Stop",
                "planned_exit_price": "Planned Exit",
                "planned_shares": "Planned Shares",
                "max_loss_dollars": "Max Loss $",
                "max_gain_dollars": "Max Gain $",
                "planned_risk_reward_ratio": "R/R",
                "planned_return_pct_to_target": "Planned Return %",
                "planned_loss_pct_to_stop": "Planned Loss %",
                "ai_plan_score": "AI Plan Score",
                "ai_plan_confidence": "AI Plan Confidence",
                "journal_warning_at_plan": "Journal Warning",
                "one_rule_to_follow_if_taken": "One Rule To Follow",
            }
            visible = filtered[[column for column in columns if column in filtered.columns]].rename(columns=columns)
            st.dataframe(visible.fillna(""), use_container_width=True, hide_index=True)

            for _, row in filtered.iterrows():
                plan_id = row["plan_id"]
                with st.expander(f"{row['ticker']} | {row['strategy_name']} | {plan_id}", expanded=False):
                    action_cols = st.columns(6)
                    if action_cols[0].button("Actualize Trade", key=f"actualize_{plan_id}", disabled=row.get("plan_status") != "Planned"):
                        st.session_state["actualize_plan_id"] = plan_id
                    if action_cols[1].button("Edit", key=f"edit_plan_{plan_id}"):
                        st.session_state["edit_plan_id"] = plan_id
                    if action_cols[2].button("Run AI Review", key=f"ai_plan_{plan_id}"):
                        with st.spinner("Running AI pre-trade review..."):
                            review_packet = {**row.to_dict(), "scanner_chart_context": journal_context_for_ticker(row["ticker"])}
                            review = run_ai_plan_review(review_packet, base_url=ollama_url, model=ollama_model, timeout=int(ai_timeout))
                            update_planned_trade(plan_id, review)
                        st.success("AI plan review saved.")
                        st.rerun()
                    if action_cols[3].button("Mark Cancelled", key=f"cancel_plan_{plan_id}"):
                        update_planned_trade(plan_id, {"plan_status": "Cancelled"})
                        st.rerun()
                    if action_cols[4].button("Mark Skipped", key=f"skip_plan_{plan_id}"):
                        update_planned_trade(plan_id, {"plan_status": "Skipped"})
                        st.rerun()
                    confirm_delete = action_cols[5].checkbox("Confirm delete", key=f"confirm_delete_{plan_id}")
                    if st.button("Delete planned trade", key=f"delete_plan_{plan_id}", disabled=not confirm_delete):
                        delete_planned_trade(plan_id)
                        st.rerun()

        edit_plan_id = st.session_state.get("edit_plan_id")
        if edit_plan_id:
            edit_rows = load_planned_trades()
            edit_rows = edit_rows[edit_rows["plan_id"].astype(str) == str(edit_plan_id)]
            if not edit_rows.empty:
                row = edit_rows.iloc[0]
                st.markdown(f"**Edit planned trade {edit_plan_id}**")
                with st.form(f"edit_plan_form_{edit_plan_id}"):
                    cols = st.columns(4)
                    edit_setup_date = cols[0].date_input("Setup date", value=pd.to_datetime(row["setup_date"], errors="coerce").date() if not pd.isna(pd.to_datetime(row["setup_date"], errors="coerce")) else datetime.now().date(), key=f"edit_date_{edit_plan_id}")
                    edit_ticker = cols[1].text_input("Ticker", value=str(row["ticker"]), key=f"edit_ticker_{edit_plan_id}").upper().strip()
                    edit_strategy = cols[2].selectbox("Strategy", STRATEGY_OPTIONS, index=STRATEGY_OPTIONS.index(row["strategy_name"]) if row["strategy_name"] in STRATEGY_OPTIONS else 0, key=f"edit_strategy_{edit_plan_id}")
                    edit_setup = cols[3].selectbox("Setup type", SETUP_TYPE_OPTIONS, index=SETUP_TYPE_OPTIONS.index(row["setup_type"]) if row["setup_type"] in SETUP_TYPE_OPTIONS else 0, key=f"edit_setup_{edit_plan_id}")
                    price_cols = st.columns(4)
                    edit_entry = price_cols[0].number_input("Planned entry price", min_value=0.0, value=float(pd.to_numeric(row["planned_entry_price"], errors="coerce") or 0), step=0.01, key=f"edit_entry_{edit_plan_id}")
                    edit_stop = price_cols[1].number_input("Planned stop loss", min_value=0.0, value=float(pd.to_numeric(row["planned_stop_loss"], errors="coerce") or 0), step=0.01, key=f"edit_stop_{edit_plan_id}")
                    edit_target = price_cols[2].number_input("Planned exit / target price", min_value=0.0, value=float(pd.to_numeric(row["planned_exit_price"], errors="coerce") or 0), step=0.01, key=f"edit_target_{edit_plan_id}")
                    edit_shares = price_cols[3].number_input("Planned shares", min_value=0, value=int(pd.to_numeric(row["planned_shares"], errors="coerce") or 0), step=1, key=f"edit_shares_{edit_plan_id}")
                    edit_cols = st.columns(2)
                    save_edit = edit_cols[0].form_submit_button("Save Edits")
                    cancel_edit = edit_cols[1].form_submit_button("Cancel Edit")
                if cancel_edit:
                    st.session_state.pop("edit_plan_id", None)
                    st.rerun()
                if save_edit:
                    errors = validate_plan(edit_ticker, edit_setup_date, edit_strategy, edit_setup, edit_entry, edit_stop, edit_target, int(edit_shares))
                    if errors:
                        for error in errors:
                            st.error(error)
                    else:
                        updates = create_planned_trade(edit_setup_date, edit_ticker, edit_strategy, edit_setup, edit_entry, edit_stop, edit_target, int(edit_shares))
                        updates.pop("plan_id", None)
                        updates.pop("created_at", None)
                        update_planned_trade(edit_plan_id, updates)
                        st.session_state.pop("edit_plan_id", None)
                        st.success("Planned trade updated.")
                        st.rerun()

        actualize_plan_id = st.session_state.get("actualize_plan_id")
        if actualize_plan_id:
            plan_rows = load_planned_trades()
            plan_rows = plan_rows[plan_rows["plan_id"].astype(str) == str(actualize_plan_id)]
            if not plan_rows.empty:
                row = plan_rows.iloc[0]
                st.markdown(f"**Actualize trade {actualize_plan_id}**")
                with st.form(f"actualize_form_{actualize_plan_id}"):
                    cols = st.columns(3)
                    actual_entry_date = cols[0].date_input("Actual entry date", value=datetime.now().date(), key=f"actual_entry_date_{actualize_plan_id}")
                    actual_buy_price = cols[1].number_input("Actual buy price", min_value=0.0, value=float(pd.to_numeric(row["planned_entry_price"], errors="coerce") or 0), step=0.01, key=f"actual_buy_{actualize_plan_id}")
                    actual_shares = cols[2].number_input("Actual shares", min_value=1, value=int(pd.to_numeric(row["planned_shares"], errors="coerce") or 1), step=1, key=f"actual_shares_{actualize_plan_id}")
                    why_entered = st.text_area("Why entered", value="", key=f"why_entered_{actualize_plan_id}")
                    actualize_cols = st.columns(2)
                    save_actualize = actualize_cols[0].form_submit_button("Save Active Trade")
                    cancel_actualize = actualize_cols[1].form_submit_button("Cancel")
                if cancel_actualize:
                    st.session_state.pop("actualize_plan_id", None)
                    st.rerun()
                if save_actualize:
                    ok, message = actualize_trade(actualize_plan_id, actual_entry_date, actual_buy_price, int(actual_shares), why_entered)
                    if ok:
                        st.session_state.pop("actualize_plan_id", None)
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

    with tabs[2]:
        st.markdown("**Active Trades**")
        planned_df = load_planned_trades()
        active = planned_df[planned_df["plan_status"] == "Active"].copy() if not planned_df.empty else pd.DataFrame()
        if active.empty:
            st.info("No active trades.")
        else:
            columns = {
                "actual_entry_date": "Actual Entry Date",
                "ticker": "Ticker",
                "strategy_name": "Strategy",
                "setup_type": "Setup Type",
                "planned_entry_price": "Planned Entry",
                "actual_buy_price": "Actual Buy",
                "entry_diff_from_plan_pct": "Entry Diff %",
                "planned_stop_loss": "Planned Stop",
                "planned_exit_price": "Planned Exit",
                "actual_shares": "Actual Shares",
                "max_loss_dollars": "Max Loss $",
                "max_gain_dollars": "Max Gain $",
                "planned_risk_reward_ratio": "R/R",
                "ai_plan_score": "AI Plan Score",
                "ai_plan_confidence": "AI Plan Confidence",
                "journal_warning_at_plan": "Journal Warning",
                "why_entered": "Why Entered",
            }
            st.dataframe(active[[column for column in columns if column in active.columns]].rename(columns=columns).fillna(""), use_container_width=True, hide_index=True)

            for _, row in active.iterrows():
                plan_id = row["plan_id"]
                with st.expander(f"Close or edit {row['ticker']} | {plan_id}", expanded=False):
                    action_cols = st.columns(2)
                    if action_cols[0].button("Close Trade", key=f"close_trade_{plan_id}"):
                        st.session_state["close_plan_id"] = plan_id
                    if action_cols[1].button("Edit Active Trade", key=f"edit_active_{plan_id}"):
                        st.session_state["actualize_plan_id"] = plan_id

        close_plan_id = st.session_state.get("close_plan_id")
        if close_plan_id:
            plan_rows = load_planned_trades()
            plan_rows = plan_rows[plan_rows["plan_id"].astype(str) == str(close_plan_id)]
            if not plan_rows.empty:
                row = plan_rows.iloc[0]
                st.markdown(f"**Close trade {close_plan_id}**")
                st.caption(
                    f"{row['ticker']} | planned entry {row['planned_entry_price']} | actual buy {row['actual_buy_price']} | "
                    f"planned stop {row['planned_stop_loss']} | planned target {row['planned_exit_price']}"
                )
                with st.form(f"close_form_{close_plan_id}"):
                    close_cols = st.columns(4)
                    actual_exit_date = close_cols[0].date_input("Actual exit date", value=datetime.now().date(), key=f"exit_date_{close_plan_id}")
                    actual_sell_price = close_cols[1].number_input("Actual sell price", min_value=0.0, value=0.0, step=0.01, key=f"sell_price_{close_plan_id}")
                    actual_exit_reason = close_cols[2].selectbox("Actual exit reason", EXIT_REASON_OPTIONS, key=f"exit_reason_{close_plan_id}")
                    followed_plan = close_cols[3].selectbox("Followed plan", ["Yes", "No", "Partially", "No Plan"], key=f"followed_plan_{close_plan_id}")
                    detail_cols = st.columns(2)
                    mistake_tag = detail_cols[0].selectbox("Mistake tag", MISTAKE_TAG_OPTIONS, key=f"mistake_tag_{close_plan_id}")
                    trade_grade_self = detail_cols[1].selectbox("Self grade", TRADE_GRADE_OPTIONS, index=6, key=f"self_grade_{close_plan_id}")
                    why_exited = st.text_area("Why exited", key=f"why_exited_{close_plan_id}")
                    lesson_learned = st.text_area("Lesson learned", key=f"lesson_{close_plan_id}")
                    notes = st.text_area("Notes", key=f"notes_{close_plan_id}")
                    screenshot = st.file_uploader("Attach daily chart screenshot", type=["png", "jpg", "jpeg", "webp"], key=f"screenshot_{close_plan_id}")
                    close_cols = st.columns(3)
                    close_trade = close_cols[0].form_submit_button("Close Trade")
                    close_with_ai = close_cols[1].form_submit_button("Close Trade and Run AI Hindsight Review")
                    cancel_close = close_cols[2].form_submit_button("Cancel")
                if cancel_close:
                    st.session_state.pop("close_plan_id", None)
                    st.rerun()
                if close_trade or close_with_ai:
                    temp_trade_id = f"pending_{close_plan_id}"
                    screenshot_path = save_uploaded_screenshot(screenshot, temp_trade_id) if screenshot is not None else ""
                    close_data = {
                        "actual_exit_date": actual_exit_date,
                        "actual_sell_price": actual_sell_price,
                        "actual_exit_reason": actual_exit_reason,
                        "followed_plan": followed_plan,
                        "why_exited": why_exited,
                        "mistake_tag": mistake_tag,
                        "lesson_learned": lesson_learned,
                        "notes": notes,
                        "trade_grade_self": trade_grade_self,
                    }
                    with st.spinner("Closing trade..."):
                        ok, message, trade_id = close_active_trade(
                            close_plan_id,
                            close_data,
                            screenshot_path=screenshot_path,
                            run_ai=close_with_ai,
                            base_url=ollama_url,
                            model=ollama_model,
                            timeout=int(ai_timeout),
                            use_cache=use_cache,
                        )
                    if ok:
                        st.session_state.pop("close_plan_id", None)
                        st.success(f"{message} Trade ID: {trade_id}")
                        st.rerun()
                    else:
                        st.error(message)

    with tabs[3]:
        st.markdown("**Closed Trade Journal**")
        trades_df = load_trades()
        if trades_df.empty:
            st.info("No closed trades yet.")
        else:
            closed_columns = [
                "trade_id",
                "plan_id",
                "actual_exit_date",
                "ticker",
                "strategy_name",
                "setup_type",
                "actual_buy_price",
                "actual_sell_price",
                "net_pnl",
                "return_pct",
                "actual_r_multiple",
                "followed_plan",
                "mistake_tag",
                "execution_score_formula",
                "ai_trade_score",
                "ai_trade_grade",
                "lesson_learned",
            ]
            visible = trades_df[[column for column in closed_columns if column in trades_df.columns]].copy()
            for column in visible.select_dtypes(include="number").columns:
                visible[column] = visible[column].round(2)
            st.dataframe(visible.fillna(""), use_container_width=True, hide_index=True)
            st.download_button("Download closed journal CSV", trades_df.to_csv(index=False), "closed_trade_journal.csv", "text/csv")

    with tabs[4]:
        st.markdown("**P&L Summary**")
        trades_df = load_trades()
        if trades_df.empty:
            st.info("No closed trades yet.")
        else:
            summary = pnl_summary(trades_df)
            keys = list(summary.keys())
            for start in range(0, len(keys), 4):
                cols = st.columns(4)
                for col, key in zip(cols, keys[start : start + 4]):
                    value = summary[key]
                    if isinstance(value, float):
                        value = round(value, 2)
                    col.metric(key, value)
            st.markdown("**Performance by Strategy**")
            st.dataframe(grouped_performance(trades_df, "strategy_name"), use_container_width=True, hide_index=True)
            st.markdown("**Performance by Ticker**")
            st.dataframe(grouped_performance(trades_df, "ticker"), use_container_width=True, hide_index=True)
            st.markdown("**Mistake Summary**")
            st.dataframe(grouped_performance(trades_df, "mistake_tag"), use_container_width=True, hide_index=True)
            st.markdown("**Followed Plan Summary**")
            st.dataframe(grouped_performance(trades_df, "followed_plan"), use_container_width=True, hide_index=True)
            st.markdown("**Recent Trades**")
            st.dataframe(trades_df.tail(20).fillna(""), use_container_width=True, hide_index=True)

    with tabs[5]:
        st.markdown("**AI Journal Coach**")
        if st.button("Run AI Journal Coach"):
            with st.spinner("Reviewing your journal patterns..."):
                coach = journal_coach_summary(base_url=ollama_url, model=ollama_model, timeout=int(ai_timeout))
            st.session_state["journal_coach_summary"] = coach
        if st.session_state.get("journal_coach_summary"):
            st.json(st.session_state["journal_coach_summary"])
        else:
            st.info("Run the coach after you have closed trades. The app still works if Ollama is offline.")

    with tabs[6]:
        st.markdown("**Personal Edge**")
        trades_df = load_trades()
        if trades_df.empty:
            st.info("No closed trades yet.")
        else:
            st.markdown("**Personal Edge by Strategy**")
            st.dataframe(grouped_performance(trades_df, "strategy_name"), use_container_width=True, hide_index=True)
            st.markdown("**Personal Edge by Setup Type**")
            st.dataframe(grouped_performance(trades_df, "setup_type"), use_container_width=True, hide_index=True)
            st.markdown("**Personal Edge by Mistake Tag**")
            st.dataframe(grouped_performance(trades_df, "mistake_tag"), use_container_width=True, hide_index=True)
            st.markdown("**Planned vs Actual Execution Summary**")
            st.dataframe(planned_vs_actual_table(trades_df), use_container_width=True, hide_index=True)

    with tabs[7]:
        st.markdown("**Planned vs Actual Review**")
        trades_df = load_trades()
        if trades_df.empty:
            st.info("No closed trades yet.")
        else:
            st.dataframe(planned_vs_actual_table(trades_df).fillna(""), use_container_width=True, hide_index=True)
            st.caption("Hindsight fields are estimated from daily OHLCV. If a daily candle touched both stop and target, intraday sequence is uncertain.")

elif page == "Chart Viewer":
    st.subheader("Chart Viewer")

    cached_tickers = get_cached_tickers()
    selected_ticker = st.session_state.get("selected_ticker", "AAPL")

    chart_source = st.radio(
        "Choose chart source",
        ["Selected ticker", "Enter ticker manually", "Choose from loaded/cache tickers"],
        index=0 if selected_ticker else 1,
    )

    if chart_source == "Choose from loaded/cache tickers" and cached_tickers:
        ticker = st.selectbox("Loaded tickers", cached_tickers)
    elif chart_source == "Selected ticker":
        ticker = st.text_input("Selected ticker", selected_ticker).upper()
    else:
        ticker = st.text_input("Enter stock ticker", selected_ticker).upper()

    period = st.selectbox("Chart period", ["3mo", "6mo", "1y", "2y"], index=2)

    support_lookback = st.slider(
        "Support/resistance lookback days",
        min_value=20,
        max_value=120,
        value=60,
        step=10,
    )

    st.sidebar.subheader("Chart overlays")

    show_ema_8 = st.sidebar.checkbox("8 EMA", value=True)
    show_sma_200 = st.sidebar.checkbox("200 SMA", value=True)
    show_support_resistance = st.sidebar.checkbox("Support / Resistance", value=True)
    show_strategy_zones = st.sidebar.checkbox("Take-profit / Sell zones", value=True)
    with st.sidebar.expander("Extra overlays", expanded=False):
        show_ema_9 = st.checkbox("EMA 9", value=False)
        show_ema_21 = st.checkbox("EMA 21", value=False)
        show_sma_50 = st.checkbox("SMA 50", value=False)
        show_ema_10 = st.checkbox("EMA 10", value=False)
        show_ema_20 = st.checkbox("EMA 20", value=False)
        show_ema_50 = st.checkbox("EMA 50", value=False)
        show_ema_100 = st.checkbox("EMA 100", value=False)
        show_ema_200 = st.checkbox("EMA 200", value=False)
        show_volume_profile = st.checkbox("Volume Profile Proxies", value=False)
        show_fvgs = st.checkbox("Estimated FVGs", value=False)
        show_order_blocks = st.checkbox("Estimated Order Blocks", value=False)
        show_vwap = st.checkbox("VWAP Proxy", value=False)
        show_liquidity_sweeps = st.checkbox("Liquidity Sweeps", value=False)

    load_chart = st.button("Load Chart") or chart_source == "Selected ticker"
    if load_chart:
        try:
            data = get_price_data(ticker, period=period, interval="1d", use_cache=use_cache)

            chart = create_trading_chart(
                data,
                ticker=ticker,
                support_lookback=support_lookback,
                show_ema_8=show_ema_8,
                show_ema_9=show_ema_9,
                show_ema_10=show_ema_10,
                show_ema_20=show_ema_20,
                show_ema_21=show_ema_21,
                show_ema_50=show_ema_50,
                show_ema_100=show_ema_100,
                show_ema_200=show_ema_200,
                show_sma_50=show_sma_50,
                show_sma_200=show_sma_200,
                show_support_resistance=show_support_resistance,
                show_volume_profile=show_volume_profile,
                show_fvgs=show_fvgs,
                show_order_blocks=show_order_blocks,
                show_vwap=show_vwap,
                show_liquidity_sweeps=show_liquidity_sweeps,
                show_strategy_zones=show_strategy_zones,
                trade_plan=st.session_state.get("selected_trade_plan"),
            )

            st.plotly_chart(chart, use_container_width=True)

            selected_strategy = st.session_state.get("selected_strategy_result")
            if selected_strategy:
                st.subheader("Selected Strategy Context")
                st.write(f"Strategy: {selected_strategy.get('strategy') or 'N/A'}")
                st.write(f"Score: {selected_strategy.get('score') or 'N/A'}")
                st.write(f"Final watch score: {selected_strategy.get('final_watch_score') or 'N/A'}")
                st.write(f"Label: {selected_strategy.get('label') or 'N/A'}")
                if selected_strategy.get("ai_action"):
                    st.write(f"AI action: {selected_strategy.get('ai_action')}")
                    st.write(f"Setup quality: {selected_strategy.get('setup_quality') or 'N/A'}")
                with st.expander("Reasons and warnings"):
                    st.write(selected_strategy.get("reasons") or "N/A")
                    st.write(selected_strategy.get("warnings") or "N/A")
                with st.expander("AI review comments"):
                    st.write(f"Entry: {selected_strategy.get('entry_comment') or 'N/A'}")
                    st.write(f"Stop: {selected_strategy.get('stop_comment') or 'N/A'}")
                    st.write(f"Target: {selected_strategy.get('target_comment') or 'N/A'}")
                    st.write(f"Confirmation needed: {selected_strategy.get('confirmation_needed') or 'N/A'}")
                    st.write(f"Main risks: {selected_strategy.get('main_risks') or 'N/A'}")
                    if show_raw_ai_json and selected_strategy.get("raw_ai_json"):
                        st.code(selected_strategy.get("raw_ai_json"), language="json")

            latest = data.iloc[-1]

            st.subheader(f"{ticker} Latest Data")
            st.write(f"Close: ${latest['close']:.2f}")
            st.write(f"High: ${latest['high']:.2f}")
            st.write(f"Low: ${latest['low']:.2f}")
            st.write(f"Volume: {latest['volume']:,.0f}")

        except Exception as error:
            st.error(f"Could not load chart for {ticker}: {error}")

    if cached_tickers:
        st.caption(f"Loaded/cache tickers available: {', '.join(cached_tickers)}")
