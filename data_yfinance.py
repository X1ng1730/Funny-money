from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from acceptance_rejection import evaluate_acceptance_rejection
from confluence import confluence_summary
from fair_value_gap import detect_fair_value_gaps
from indicators import add_indicators, latest_complete_row
from liquidity import detect_liquidity_sweeps
from order_blocks import detect_order_blocks
from support_resistance import calculate_support_resistance_levels, percent_distance
from volume_profile import calculate_volume_profile
from vwap import vwap_features

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
WATCHLIST_MARKET_CACHE = DATA_DIR / "watchlist_market_cache.csv"
try:
    yfinance_cache_dir = Path("data") / "yfinance_cache"
    yfinance_cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(yfinance_cache_dir.resolve()))
except Exception:
    pass

YFINANCE_SYMBOL_MAP = {
    "KIOXIA": "285A.T",
}


def clear_watchlist_market_cache() -> None:
    if WATCHLIST_MARKET_CACHE.exists():
        WATCHLIST_MARKET_CACHE.unlink()


def _load_watchlist_market_cache(tickers: list[str], max_age_minutes: int = 60) -> pd.DataFrame | None:
    if not WATCHLIST_MARKET_CACHE.exists():
        return None
    try:
        cache = pd.read_csv(WATCHLIST_MARKET_CACHE)
        if cache.empty or "ticker" not in cache.columns or "cache_timestamp" not in cache.columns:
            return None
        cache_time = pd.to_datetime(cache["cache_timestamp"].iloc[0], errors="coerce")
        if pd.isna(cache_time):
            return None
        if datetime.now() - cache_time.to_pydatetime() > timedelta(minutes=max_age_minutes):
            return None
        requested = {ticker.upper().strip() for ticker in tickers}
        cached = set(cache["ticker"].astype(str).str.upper().str.strip())
        if not requested.issubset(cached):
            return None
        subset = cache[cache["ticker"].astype(str).str.upper().isin(requested)].copy()
        return subset.drop(columns=["cache_timestamp"], errors="ignore")
    except Exception:
        return None


def _save_watchlist_market_cache(df: pd.DataFrame) -> None:
    if df.empty:
        return
    cache = df.copy()
    cache["cache_timestamp"] = datetime.now().isoformat(timespec="seconds")
    cache.to_csv(WATCHLIST_MARKET_CACHE, index=False)


def get_price_data(
    ticker: str, period: str = "6mo", interval: str = "1d", use_cache: bool = True
) -> pd.DataFrame:
    ticker = ticker.upper().strip()

    if not ticker:
        raise ValueError("Ticker cannot be empty")

    cache_file = CACHE_DIR / f"{ticker}_{period}_{interval}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        required_columns = {"open", "high", "low", "close", "volume"}
        if required_columns.issubset(df.columns) and not df.empty:
            return df

        cache_file.unlink()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as error:
        raise RuntimeError(f"Could not download data for {ticker}: {error}") from error

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # yfinance sometimes returns multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing price columns for {ticker}: {missing_columns}")

    df = df[required_columns].dropna()

    if df.empty:
        raise ValueError(f"No complete OHLCV rows returned for {ticker}")

    df.to_csv(cache_file)

    return df


def get_multiple_price_data(
    tickers: list[str],
    period: str = "6mo",
    interval: str = "1d",
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            results[ticker] = get_price_data(ticker, period, interval, use_cache)
        except Exception as error:
            print(f"Could not load {ticker}: {error}")

    return results


def _safe_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_date(value) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, pd.DataFrame) and not value.empty:
            value = value.index[0]
        if isinstance(value, dict):
            value = next(iter(value.values()), None)
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        parsed = pd.to_datetime(value)
        if pd.isna(parsed):
            return None
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return None


def _market_cap_category(market_cap: float | None) -> str:
    if market_cap is None:
        return "N/A"
    if market_cap < 2_000_000_000:
        return "Small / speculative"
    if market_cap < 10_000_000_000:
        return "Small-mid growth"
    if market_cap < 50_000_000_000:
        return "Mid-large"
    if market_cap < 200_000_000_000:
        return "Large cap"
    return "Mega cap"


def _volatility_rating(atr_pct: float | None) -> str:
    if atr_pct is None:
        return "N/A"
    if atr_pct < 2:
        return "Low"
    if atr_pct < 4:
        return "Medium"
    if atr_pct < 7:
        return "High"
    return "Very High"


def _trend_status(row: pd.Series, previous: pd.Series | None) -> str:
    price = _safe_float(row.get("close"))
    ema_8 = _safe_float(row.get("EMA_8"))
    ema_21 = _safe_float(row.get("EMA_21"))
    sma_50 = _safe_float(row.get("SMA_50"))
    sma_200 = _safe_float(row.get("SMA_200"))
    if None in [price, ema_8, ema_21, sma_50, sma_200]:
        return "Data Insufficient"

    crossed_8 = False
    crossed_200 = False
    if previous is not None:
        prev_price = _safe_float(previous.get("close"))
        prev_ema_8 = _safe_float(previous.get("EMA_8"))
        prev_sma_200 = _safe_float(previous.get("SMA_200"))
        crossed_8 = prev_price is not None and prev_ema_8 is not None and prev_price <= prev_ema_8 and price > ema_8
        crossed_200 = prev_price is not None and prev_sma_200 is not None and prev_price <= prev_sma_200 and price > sma_200

    if price > ema_8 > ema_21 > sma_50 and price > sma_200:
        return "Strong Uptrend"
    if price > ema_21 and price > sma_50 and price > sma_200:
        return "Uptrend"
    if price < ema_21 and price > sma_50 and price > sma_200:
        return "Pullback in Uptrend"
    if crossed_8 or crossed_200:
        return "Reclaiming Trend"
    if price < sma_200 and ema_8 < sma_200:
        return "Long-Term Weak"
    if price < sma_50 or price < sma_200:
        return "Breakdown Risk"
    return "Rangebound"


def _setup_type(row: pd.Series, levels: dict) -> str:
    price = _safe_float(row.get("close"))
    ema_8 = _safe_float(row.get("EMA_8"))
    ema_21 = _safe_float(row.get("EMA_21"))
    sma_50 = _safe_float(row.get("SMA_50"))
    sma_200 = _safe_float(row.get("SMA_200"))
    rvol = _safe_float(row.get("relative_volume"))
    return_5d = _safe_float(row.get("return_5d"))
    rsi = _safe_float(row.get("RSI_14"))
    resistance = levels.get("nearest_resistance")
    if price is None:
        return "No Clear Setup"

    distance_to_resistance = percent_distance(price, resistance)
    above_recent_resistance = resistance is not None and 0 <= (price - resistance) / resistance * 100 <= 5

    if price is not None and ema_8 is not None and (price > ema_8 * 1.08 or (rsi is not None and rsi > 70)):
        return "Extended"
    if price is not None and ((sma_50 is not None and price < sma_50) or (sma_200 is not None and price < sma_200)) and return_5d is not None and return_5d < 0:
        return "Breakdown Risk"
    if distance_to_resistance is not None and -3 <= distance_to_resistance <= 3 and rvol is not None and rvol > 1.2:
        return "Breakout Watch"
    if above_recent_resistance and rvol is not None and rvol > 1.2:
        return "Recent Breakout"
    if return_5d is not None and return_5d > 0.05 and ema_8 is not None and price > ema_8 and rvol is not None and rvol > 1.2:
        return "Momentum Continuation"
    if (ema_21 is not None and abs(price - ema_21) / price <= 0.03) or (sma_50 is not None and abs(price - sma_50) / price <= 0.03):
        if sma_200 is not None and price > sma_200:
            return "Pullback Watch"
    return "No Clear Setup"


def _latest_headline(ticker: yf.Ticker) -> str | None:
    try:
        news = ticker.news
        if news:
            return news[0].get("title")
    except Exception:
        return None
    return None


def _metadata_for_ticker(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    info = {}
    try:
        info = ticker.get_info() or {}
    except Exception:
        try:
            info = ticker.info or {}
        except Exception:
            info = {}

    analyst_target = _safe_float(info.get("targetMeanPrice") or info.get("targetMedianPrice"))
    current = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    analyst_upside = None
    if analyst_target is not None and current:
        analyst_upside = (analyst_target - current) / current * 100

    earnings_date = None
    try:
        earnings_date = _safe_date(ticker.get_earnings_dates(limit=1))
    except Exception:
        earnings_date = _safe_date(info.get("earningsDate"))

    return {
        "market_cap": _safe_float(info.get("marketCap")),
        "pe_ratio": _safe_float(info.get("trailingPE")),
        "forward_pe": _safe_float(info.get("forwardPE")),
        "beta": _safe_float(info.get("beta")),
        "analyst_target_upside_pct": analyst_upside,
        "recommendation": info.get("recommendationKey") or info.get("recommendationMean"),
        "next_earnings_date": earnings_date,
        "latest_headline": _latest_headline(ticker),
    }


def _risk_flags(row: dict, manual_catalyst: str = "") -> str:
    flags = []
    price = row.get("current_price")
    ema_8 = row.get("ema_8")
    sma_50 = row.get("sma_50")
    sma_200 = row.get("sma_200")
    atr_pct = row.get("atr_pct")
    rsi = row.get("rsi_14")
    rvol = row.get("relative_volume")
    dollar_volume = row.get("dollar_volume")
    earnings = row.get("next_earnings_date")
    if row.get("data_status") != "OK":
        flags.append("Data Missing")
    if row.get("data_quality_score") is not None and row.get("data_quality_score") < 70:
        flags.append("Data Quality Warning")
    if earnings:
        try:
            days = (pd.to_datetime(earnings).date() - pd.Timestamp.now().date()).days
            if 0 <= days <= 7:
                flags.append("Earnings Soon")
            if 0 <= days <= 2:
                flags.append("Earnings Very Soon")
        except Exception:
            pass
    if atr_pct is not None and atr_pct > 7:
        flags.append("Very High ATR")
    elif atr_pct is not None and atr_pct > 4:
        flags.append("High ATR")
    if price is not None and ema_8 is not None and price > ema_8 * 1.08:
        flags.append("Extended Above 8/9 EMA")
    elif rsi is not None and rsi > 70:
        flags.append("Extended Above 8/9 EMA")
    if rsi is not None and rsi > 75:
        flags.append("RSI Overbought")
    if price is not None and sma_50 is not None and price < sma_50:
        flags.append("Below 50D")
    if price is not None and sma_200 is not None and price < sma_200:
        flags.append("Below 200D")
    if dollar_volume is not None and dollar_volume < 25_000_000:
        flags.append("Low Liquidity")
    if rvol is not None and rvol < 0.8:
        flags.append("Low RVOL")
    if rvol is not None and rvol > 2.0:
        flags.append("Unusual RVOL")
    elif rvol is not None and rvol > 1.2:
        flags.append("Active RVOL")
    if row.get("resistance_breakout_status") == "Failed Breakout":
        flags.append("Gap Failed")
    if row.get("breakout_acceptance_status") == "failed":
        flags.append("Breakout Rejected")
    if row.get("breakout_sweep_failure_flag"):
        flags.append("Liquidity Sweep Above High")
    if row.get("bullish_liquidity_sweep_detected"):
        flags.append("Bullish Liquidity Sweep")
    if row.get("bullish_fvg_active"):
        flags.append("Estimated Bullish FVG")
    if row.get("bullish_order_block_active"):
        flags.append("Estimated Bullish Order Block")
    if row.get("vwap_status") == "intraday data unavailable":
        flags.append("VWAP Proxy Only")
    if row.get("distance_to_nearest_resistance_pct") is not None and abs(row["distance_to_nearest_resistance_pct"]) <= 2:
        flags.append("Near Major Resistance")
    if row.get("risk_reward_target_1") is not None and row["risk_reward_target_1"] < 1.5:
        flags.append("Poor Risk/Reward")
    if not str(manual_catalyst).strip() and not row.get("latest_headline"):
        flags.append("No Catalyst")
    if row.get("relative_strength_spy_1m") is not None and row.get("relative_strength_spy_1m") < 0:
        flags.append("Weak Relative Strength")
    if row.get("latest_headline") and row.get("return_1d_pct") is not None and abs(row["return_1d_pct"]) > 5:
        flags.append("News-Driven")
    if str(manual_catalyst).strip():
        flags.append("Manual Catalyst")
    return ", ".join(flags)


def _data_quality_score(row: dict, history: pd.DataFrame, metadata: dict) -> int:
    score = 100
    if history.empty:
        return 0
    if len(history) < 200:
        score -= 20
    if len(history) < 50:
        score -= 25
    for column in ["current_price", "volume", "relative_volume", "atr_pct", "rsi_14"]:
        if row.get(column) is None:
            score -= 8
    if metadata.get("market_cap") is None:
        score -= 8
    if metadata.get("next_earnings_date") is None:
        score -= 4
    if metadata.get("latest_headline") is None:
        score -= 3
    if row.get("vwap_status") == "intraday data unavailable":
        score -= 2
    return max(0, min(100, score))


def _readiness_score(row: dict, manual_catalyst: str = "") -> tuple[int, str]:
    score = 0
    trend = row.get("trend_status")
    if trend in {"Strong Uptrend", "Uptrend"}:
        score += 20
    elif trend in {"Pullback in Uptrend", "Reclaiming Trend"}:
        score += 14
    elif trend == "Rangebound":
        score += 8

    rvol = row.get("relative_volume")
    if rvol is not None:
        score += 15 if rvol >= 1.2 else 9 if rvol >= 0.8 else 3

    rs_values = [row.get("relative_strength_spy_1m"), row.get("relative_strength_qqq_1m")]
    if any(value is not None and value > 0 for value in rs_values):
        score += 15
    elif any(value is not None for value in rs_values):
        score += 7

    support = row.get("distance_to_nearest_support_pct")
    resistance = row.get("distance_to_nearest_resistance_pct")
    if support is not None and resistance is not None and support <= 8 and resistance >= -3:
        score += 20
    elif resistance is not None and resistance >= -5:
        score += 12

    if str(manual_catalyst).strip() or row.get("latest_headline"):
        score += 10

    dollar_volume = row.get("dollar_volume")
    if dollar_volume is not None:
        score += 10 if dollar_volume >= 25_000_000 else 3

    atr_pct = row.get("atr_pct")
    if atr_pct is not None:
        score += 10 if atr_pct <= 7 else 4

    score = max(0, min(100, score))
    if score >= 80:
        label = "High priority"
    elif score >= 65:
        label = "Watch closely"
    elif score >= 50:
        label = "Neutral / developing"
    elif score >= 35:
        label = "Weak setup"
    else:
        label = "Avoid for now"
    return score, label


def _relative_strength(df: pd.DataFrame, benchmark: pd.DataFrame | None, days: int = 20) -> float | None:
    if benchmark is None or df.empty or benchmark.empty:
        return None
    ticker_return = _safe_float(df["close"].pct_change(days).iloc[-1])
    benchmark_return = _safe_float(benchmark["close"].pct_change(days).iloc[-1])
    if ticker_return is None or benchmark_return is None:
        return None
    return (ticker_return - benchmark_return) * 100


def _crossed_above(df: pd.DataFrame, column: str, days: int) -> bool:
    recent = df.dropna(subset=["close", column]).tail(days + 1)
    if len(recent) < 2:
        return False
    previous = recent.iloc[:-1]
    latest = recent.iloc[-1]
    return bool((previous["close"] <= previous[column]).any() and latest["close"] > latest[column])


def _market_regime(benchmarks: dict[str, pd.DataFrame | None]) -> str:
    bullish = 0
    bearish = 0
    for df in benchmarks.values():
        if df is None or df.empty:
            continue
        enriched = add_indicators(df)
        latest = latest_complete_row(enriched)
        close = _safe_float(latest.get("close"))
        ema_21 = _safe_float(latest.get("EMA_21"))
        sma_50 = _safe_float(latest.get("SMA_50"))
        if close is not None and ema_21 is not None and sma_50 is not None and close > ema_21 and close > sma_50:
            bullish += 1
        elif close is not None and sma_50 is not None and close < sma_50:
            bearish += 1
    if bullish >= 2:
        return "Risk-On"
    if bearish >= 2:
        return "Risk-Off"
    return "Mixed"


def build_market_snapshot(ticker: str, history: pd.DataFrame, metadata: dict, benchmarks: dict[str, pd.DataFrame] | None = None) -> dict:
    benchmarks = benchmarks or {}
    if history.empty:
        return {"ticker": ticker, "data_status": "Data unavailable"}

    df = add_indicators(history)
    latest = latest_complete_row(df)
    previous = df.dropna(subset=["close"]).iloc[-2] if len(df.dropna(subset=["close"])) > 1 else None
    levels = calculate_support_resistance_levels(df)
    profile = calculate_volume_profile(df, lookback=60)
    profile.pop("volume_by_price", None)
    weekly_profile = calculate_volume_profile(
        df.resample("W").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(),
        lookback=52,
        bins=40,
    )
    weekly_profile.pop("volume_by_price", None)
    fvg = detect_fair_value_gaps(df, profile.get("vp_nearest_lvn_below") or profile.get("vp_nearest_lvn_above"))
    order_block = detect_order_blocks(df, profile.get("vp_nearest_hvn_below") or profile.get("vp_nearest_hvn_above"))
    sweeps = detect_liquidity_sweeps(df)
    acceptance = evaluate_acceptance_rejection(latest, levels.get("breakout_level"), "breakout level")
    vwap = vwap_features(df, intraday_available=False)
    price = _safe_float(latest.get("close"))

    row = {
        "ticker": ticker,
        "data_status": "OK",
        "current_price": price,
        "return_1d_pct": _safe_float(latest.get("return_1d") * 100),
        "return_5d_pct": _safe_float(latest.get("return_5d") * 100),
        "return_1m_pct": _safe_float(latest.get("return_1m") * 100),
        "return_3m_pct": _safe_float(latest.get("return_3m") * 100),
        "volume": _safe_float(latest.get("volume")),
        "average_volume_20d": _safe_float(latest.get("volume_avg_20")),
        "relative_volume": _safe_float(latest.get("relative_volume")),
        "dollar_volume": _safe_float(latest.get("dollar_volume_avg_20")),
        "market_cap": metadata.get("market_cap"),
        "market_cap_category": _market_cap_category(metadata.get("market_cap")),
        "pe_ratio": metadata.get("pe_ratio"),
        "forward_pe": metadata.get("forward_pe"),
        "beta": metadata.get("beta"),
        "atr_pct": _safe_float(latest.get("ATR_pct")),
        "volatility_rating": _volatility_rating(_safe_float(latest.get("ATR_pct"))),
        "ema_8": _safe_float(latest.get("EMA_8")),
        "ema_9": _safe_float(latest.get("EMA_9")),
        "ema_21": _safe_float(latest.get("EMA_21")),
        "sma_50": _safe_float(latest.get("SMA_50")),
        "sma_200": _safe_float(latest.get("SMA_200")),
        "atr_14": _safe_float(latest.get("ATR_14")),
        "rsi_14": _safe_float(latest.get("RSI_14")),
        "gap_pct": _safe_float(latest.get("gap_pct")),
        "close_vs_previous_close_pct": _safe_float(latest.get("close_vs_previous_close_pct")),
        "previous_high": _safe_float(previous.get("high")) if previous is not None else None,
        "previous_low": _safe_float(previous.get("low")) if previous is not None else None,
        "previous_close": _safe_float(previous.get("close")) if previous is not None else None,
        "trend_status": _trend_status(latest, previous),
        "setup_type": _setup_type(latest, levels),
        "high_20d": _safe_float(latest.get("high_20d")),
        "high_50d": _safe_float(latest.get("high_50d")),
        "high_100d": _safe_float(latest.get("high_100d")),
        "low_20d": _safe_float(latest.get("low_20d")),
        "low_50d": _safe_float(latest.get("low_50d")),
        "low_100d": _safe_float(latest.get("low_100d")),
        "high_52w": _safe_float(latest.get("high_52w")),
        "low_52w": _safe_float(latest.get("low_52w")),
        "distance_to_52w_high_pct": percent_distance(price, _safe_float(latest.get("high_52w"))),
        "distance_to_52w_low_pct": percent_distance(price, _safe_float(latest.get("low_52w"))),
        "distance_to_nearest_support_pct": percent_distance(price, levels.get("nearest_support")),
        "distance_to_nearest_resistance_pct": percent_distance(price, levels.get("nearest_resistance")),
        "distance_from_8ema_pct": _safe_float(latest.get("distance_from_8ema_pct")),
        "distance_from_9ema_pct": _safe_float(latest.get("distance_from_9ema_pct")),
        "distance_from_21ema_pct": _safe_float(latest.get("distance_from_21ema_pct")),
        "distance_from_50sma_pct": _safe_float(latest.get("distance_from_50sma_pct")),
        "distance_from_200sma_pct": _safe_float(latest.get("distance_from_200sma_pct")),
        "support_20d": levels.get("support_20d"),
        "support_50d": levels.get("support_50d"),
        "support_100d": levels.get("support_100d"),
        "support_level": levels.get("support_level"),
        "resistance_20d": levels.get("resistance_20d"),
        "resistance_50d": levels.get("resistance_50d"),
        "resistance_100d": levels.get("resistance_100d"),
        "breakout_level": levels.get("breakout_level"),
        "nearest_support": levels.get("nearest_support"),
        "nearest_resistance": levels.get("nearest_resistance"),
        "resistance_breakout_status": levels.get("resistance_breakout_status"),
        "support_hold_status": levels.get("support_hold_status"),
        "crossed_above_8ema_3d": _crossed_above(df, "EMA_8", 3),
        "crossed_above_21ema_5d": _crossed_above(df, "EMA_21", 5),
        "crossed_above_200sma_10d": _crossed_above(df, "SMA_200", 10),
        "analyst_target_upside_pct": metadata.get("analyst_target_upside_pct"),
        "recommendation": metadata.get("recommendation"),
        "next_earnings_date": metadata.get("next_earnings_date"),
        "latest_headline": metadata.get("latest_headline"),
        "relative_strength_spy_5d": _relative_strength(df, benchmarks.get("SPY"), 5),
        "relative_strength_qqq_5d": _relative_strength(df, benchmarks.get("QQQ"), 5),
        "relative_strength_spy_1m": _relative_strength(df, benchmarks.get("SPY")),
        "relative_strength_qqq_1m": _relative_strength(df, benchmarks.get("QQQ")),
        "market_regime": _market_regime(benchmarks),
    }
    row.update(profile)
    row.update(
        {
            "weekly_vp_poc": weekly_profile.get("vp_poc"),
            "weekly_value_area_high": weekly_profile.get("vp_value_area_high"),
            "weekly_value_area_low": weekly_profile.get("vp_value_area_low"),
            "weekly_nearest_hvn_above": weekly_profile.get("vp_nearest_hvn_above"),
            "weekly_nearest_hvn_below": weekly_profile.get("vp_nearest_hvn_below"),
            "weekly_nearest_lvn_above": weekly_profile.get("vp_nearest_lvn_above"),
            "weekly_acceptance_status": weekly_profile.get("vp_current_location"),
            "weekly_resistance_overhead_pct": percent_distance(price, weekly_profile.get("vp_nearest_hvn_above")),
            "weekly_support_below_pct": percent_distance(price, weekly_profile.get("vp_nearest_hvn_below")),
        }
    )
    row.update(fvg)
    row.update(order_block)
    row.update(sweeps)
    row.update(acceptance)
    row.update(vwap)
    row.update(confluence_summary(pd.Series(row)))
    row["advanced_technical_score"] = _advanced_technical_score(row)
    row["breakout_quality_score"] = _breakout_quality_score(row)
    row["liquidity_sweep_status"] = "bullish sweep/reclaim" if row.get("bullish_liquidity_sweep_detected") else "none"
    row["fvg_lvn_status"] = "active FVG + LVN" if row.get("bullish_fvg_active") and row.get("bullish_fvg_overlaps_lvn") else "active FVG" if row.get("bullish_fvg_active") else "none"
    row["ob_hvn_status"] = "active OB + HVN" if row.get("bullish_order_block_active") and row.get("bullish_order_block_overlaps_hvn") else "active OB" if row.get("bullish_order_block_active") else "none"
    row["volume_profile_location"] = row.get("vp_current_location")
    row["key_level_status"] = row.get("key_level_status")
    row["trade_plan_type"] = "proxy levels available" if row.get("confluence_score", 0) > 0 else "standard technical levels"
    row["data_quality_score"] = _data_quality_score(row, history, metadata)
    row["risk_flags"] = _risk_flags(row)
    row["trade_readiness_score"], row["trade_readiness_label"] = _readiness_score(row)
    return row


def _advanced_technical_score(row: dict) -> int:
    score = 0
    score += min(20, (row.get("sweep_reclaim_strength_score") or 0) * 0.20)
    score += min(15, (row.get("acceptance_score") or 0) * 0.15)
    score += min(15, (row.get("bullish_fvg_quality_score") or 0) * 0.15)
    score += min(15, (row.get("bullish_order_block_quality_score") or 0) * 0.15)
    score += 10 if row.get("vp_current_location") in {"above value", "inside value"} else 0
    score += 10 if row.get("price_above_vwap") else 0
    score += min(15, (row.get("confluence_score") or 0) * 0.15)
    if row.get("rejected_at_resistance") or row.get("breakout_sweep_failure_flag"):
        score -= 15
    return max(0, min(100, round(score)))


def _breakout_quality_score(row: dict) -> int:
    score = 0
    if row.get("accepted_above_breakout_level"):
        score += 35
    if (row.get("relative_volume") or 0) > 1.2:
        score += 15
    if row.get("vp_nearest_lvn_above") is not None and row.get("vp_nearest_hvn_above") is not None:
        score += 15
    if row.get("price_above_vwap"):
        score += 10
    if row.get("support_hold_status") == "Holding Support":
        score += 10
    if row.get("rejected_at_resistance"):
        score -= 25
    return max(0, min(100, round(score)))


def get_watchlist_market_data(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
    use_persistent_cache: bool = True,
    cache_max_age_minutes: int = 60,
) -> pd.DataFrame:
    unique_tickers = sorted({ticker.upper().strip() for ticker in tickers if str(ticker).strip()})
    if not unique_tickers:
        return pd.DataFrame()

    if use_persistent_cache:
        cached = _load_watchlist_market_cache(unique_tickers, cache_max_age_minutes)
        if cached is not None:
            return cached

    symbols = {ticker: YFINANCE_SYMBOL_MAP.get(ticker, ticker) for ticker in unique_tickers}
    histories: dict[str, pd.DataFrame] = {}
    download_symbols = sorted(set(symbols.values()) | {"SPY", "QQQ"})

    try:
        batch = yf.download(
            " ".join(download_symbols),
            period=period,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=False,
        )
    except Exception:
        batch = pd.DataFrame()

    for ticker, symbol in symbols.items():
        try:
            if isinstance(batch.columns, pd.MultiIndex):
                raw = batch[symbol].copy()
            else:
                raw = batch.copy() if len(download_symbols) == 1 else pd.DataFrame()
            raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            histories[ticker] = raw[["open", "high", "low", "close", "volume"]].dropna(how="all")
        except Exception:
            try:
                histories[ticker] = get_price_data(symbol, period=period, interval=interval, use_cache=False)
            except Exception:
                histories[ticker] = pd.DataFrame()

    benchmarks = {}
    for benchmark in ["SPY", "QQQ"]:
        try:
            if isinstance(batch.columns, pd.MultiIndex):
                raw = batch[benchmark].copy()
            else:
                raw = pd.DataFrame()
            raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            benchmarks[benchmark] = raw[["open", "high", "low", "close", "volume"]].dropna(how="all")
        except Exception:
            benchmarks[benchmark] = None

    rows = []
    for ticker in unique_tickers:
        history = histories.get(ticker, pd.DataFrame())
        if history.empty:
            rows.append({"ticker": ticker, "data_status": "Data unavailable", "risk_flags": "Data Missing"})
            continue
        metadata = _metadata_for_ticker(symbols[ticker])
        try:
            rows.append(build_market_snapshot(ticker, history, metadata, benchmarks))
        except Exception as error:
            rows.append({"ticker": ticker, "data_status": f"Error: {error}", "risk_flags": "Data Missing"})

    result = pd.DataFrame(rows)
    if use_persistent_cache:
        _save_watchlist_market_cache(result)
    return result
