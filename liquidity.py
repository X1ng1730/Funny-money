import pandas as pd


def _pivot_lows(df: pd.DataFrame, window: int = 3) -> list[tuple[str, float]]:
    levels: list[tuple[str, float]] = []
    lows = df["low"].dropna()
    for idx in range(window, len(lows) - window):
        current = lows.iloc[idx]
        if current <= lows.iloc[idx - window:idx].min() and current <= lows.iloc[idx + 1:idx + window + 1].min():
            levels.append(("pivot low", float(current)))
    return levels[-5:]


def _pivot_highs(df: pd.DataFrame, window: int = 3) -> list[tuple[str, float]]:
    levels: list[tuple[str, float]] = []
    highs = df["high"].dropna()
    for idx in range(window, len(highs) - window):
        current = highs.iloc[idx]
        if current >= highs.iloc[idx - window:idx].max() and current >= highs.iloc[idx + 1:idx + window + 1].max():
            levels.append(("pivot high", float(current)))
    return levels[-5:]


def detect_liquidity_sweeps(df: pd.DataFrame, sweep_buffer: float = 0.001) -> dict:
    recent = df.dropna(subset=["open", "high", "low", "close"]).tail(80)
    if len(recent) < 5:
        return {
            "bullish_liquidity_sweep_detected": False,
            "swept_level": None,
            "swept_level_type": None,
            "sweep_reclaim_strength_score": 0,
            "sweep_wick_pct": None,
            "sweep_volume_confirmation": False,
            "bearish_sweep_of_high_detected": False,
            "breakout_sweep_failure_flag": False,
        }

    latest = recent.iloc[-1]
    previous = recent.iloc[-2]
    candle_range = latest["high"] - latest["low"]
    lower_wick_pct = 0 if candle_range <= 0 else (min(latest["open"], latest["close"]) - latest["low"]) / candle_range * 100
    upper_wick_pct = 0 if candle_range <= 0 else (latest["high"] - max(latest["open"], latest["close"])) / candle_range * 100
    close_position = 0 if candle_range <= 0 else (latest["close"] - latest["low"]) / candle_range
    rvol = latest.get("relative_volume")
    volume_ok = bool(pd.notna(rvol) and rvol >= 1.0)

    low_levels = [("previous day low", float(previous["low"]))]
    high_levels = [("previous day high", float(previous["high"]))]
    for lookback in [5, 20, 50]:
        if len(recent) > lookback:
            low_levels.append((f"{lookback}D low", float(recent["low"].iloc[:-1].tail(lookback).min())))
            high_levels.append((f"{lookback}D high", float(recent["high"].iloc[:-1].tail(lookback).max())))
    low_levels.extend(_pivot_lows(recent.iloc[:-1]))
    high_levels.extend(_pivot_highs(recent.iloc[:-1]))

    swept = [(name, level) for name, level in low_levels if latest["low"] < level * (1 - sweep_buffer) and latest["close"] > level]
    swept_high = [(name, level) for name, level in high_levels if latest["high"] > level * (1 + sweep_buffer) and latest["close"] < level]

    strength = 0
    if swept:
        strength += 30
    if lower_wick_pct > 35:
        strength += 25
    if close_position >= 0.5:
        strength += 20
    if volume_ok:
        strength += 15
    if pd.notna(rvol) and rvol >= 1.5:
        strength += 10

    selected = swept[-1] if swept else (None, None)
    return {
        "bullish_liquidity_sweep_detected": bool(swept),
        "swept_level": selected[1],
        "swept_level_type": selected[0],
        "sweep_reclaim_strength_score": min(100, strength),
        "sweep_wick_pct": round(lower_wick_pct, 2),
        "sweep_volume_confirmation": volume_ok,
        "bearish_sweep_of_high_detected": bool(swept_high),
        "breakout_sweep_failure_flag": bool(swept_high and upper_wick_pct > 35),
    }
