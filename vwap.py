import pandas as pd


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    typical = (result["high"] + result["low"] + result["close"]) / 3
    cumulative_volume = result["volume"].cumsum()
    result["VWAP"] = (typical * result["volume"]).cumsum() / cumulative_volume
    return result


def vwap_features(df: pd.DataFrame, intraday_available: bool = False) -> dict:
    if df.empty:
        return _empty(intraday_available)
    enriched = add_vwap(df.dropna(subset=["high", "low", "close", "volume"]).tail(80))
    if enriched.empty or enriched["VWAP"].dropna().empty:
        return _empty(intraday_available)
    latest = enriched.iloc[-1]
    previous = enriched.iloc[-2] if len(enriched) > 1 else latest
    vwap = float(latest["VWAP"])
    close = float(latest["close"])
    slope = float(enriched["VWAP"].tail(5).diff().mean()) if len(enriched) >= 5 else 0
    return {
        "intraday_vwap": round(vwap, 2),
        "price_above_vwap": close > vwap,
        "vwap_slope": round(slope, 4),
        "vwap_reclaim_detected": previous["close"] <= previous["VWAP"] and close > vwap,
        "vwap_rejection_detected": latest["high"] > vwap and close < vwap,
        "distance_to_vwap_pct": round((close - vwap) / close * 100, 2) if close else None,
        "vwap_status": "above VWAP proxy" if close > vwap else "below VWAP proxy",
        "intraday_vwap_available": intraday_available,
    }


def _empty(intraday_available: bool) -> dict:
    return {
        "intraday_vwap": None,
        "price_above_vwap": False,
        "vwap_slope": None,
        "vwap_reclaim_detected": False,
        "vwap_rejection_detected": False,
        "distance_to_vwap_pct": None,
        "vwap_status": "intraday data unavailable" if not intraday_available else "VWAP unavailable",
        "intraday_vwap_available": intraday_available,
    }
