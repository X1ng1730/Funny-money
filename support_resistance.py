import pandas as pd


def _pivot_levels(series: pd.Series, direction: str, window: int = 2) -> list[float]:
    levels: list[float] = []
    values = series.dropna()
    if len(values) < window * 2 + 1:
        return levels
    for index in range(window, len(values) - window):
        current = values.iloc[index]
        left = values.iloc[index - window:index]
        right = values.iloc[index + 1:index + window + 1]
        if direction == "high" and current >= left.max() and current >= right.max():
            levels.append(float(current))
        if direction == "low" and current <= left.min() and current <= right.min():
            levels.append(float(current))
    return levels[-8:]


def calculate_support_resistance_levels(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "support_20d": None,
            "support_50d": None,
            "support_100d": None,
            "resistance_20d": None,
            "resistance_50d": None,
            "resistance_100d": None,
            "nearest_support": None,
            "nearest_resistance": None,
            "breakout_level": None,
            "support_level": None,
            "resistance_breakout_status": "Data Insufficient",
            "support_hold_status": "Data Insufficient",
        }

    latest_price = float(df["close"].dropna().iloc[-1]) if not df["close"].dropna().empty else None
    lows_20 = df["low"].tail(20).dropna()
    lows_50 = df["low"].tail(50).dropna()
    lows_100 = df["low"].tail(100).dropna()
    highs_20 = df["high"].tail(20).dropna()
    highs_50 = df["high"].tail(50).dropna()
    highs_100 = df["high"].tail(100).dropna()

    support_20d = float(lows_20.min()) if not lows_20.empty else None
    support_50d = float(lows_50.min()) if not lows_50.empty else None
    support_100d = float(lows_100.min()) if not lows_100.empty else None
    resistance_20d = float(highs_20.max()) if not highs_20.empty else None
    resistance_50d = float(highs_50.max()) if not highs_50.empty else None
    resistance_100d = float(highs_100.max()) if not highs_100.empty else None
    pivot_highs = _pivot_levels(df["high"].tail(120), "high")
    pivot_lows = _pivot_levels(df["low"].tail(120), "low")

    latest = df.iloc[-1]
    ma_supports = []
    for column in ["EMA_8", "EMA_9", "EMA_21", "SMA_50", "SMA_200"]:
        value = latest.get(column)
        if value is not None and pd.notna(value):
            ma_supports.append(float(value))

    supports = [value for value in [support_20d, support_50d, support_100d] + pivot_lows + ma_supports if value is not None]
    resistances = [value for value in [resistance_20d, resistance_50d, resistance_100d] + pivot_highs if value is not None]

    nearest_support = None
    nearest_resistance = None
    breakout_level = None
    resistance_breakout_status = "Data Insufficient"
    support_hold_status = "Data Insufficient"
    if latest_price is not None:
        below_price = [value for value in supports if value <= latest_price]
        above_price = [value for value in resistances if value >= latest_price]
        nearest_support = max(below_price) if below_price else (max(supports) if supports else None)
        nearest_resistance = min(above_price) if above_price else (min(resistances) if resistances else None)
        previous_resistance_candidates = []
        for lookback in [20, 50, 100]:
            previous_highs = df["high"].iloc[:-1].tail(lookback).dropna()
            if not previous_highs.empty:
                previous_resistance_candidates.append(float(previous_highs.max()))
        breakout_level = max(previous_resistance_candidates) if previous_resistance_candidates else nearest_resistance
        if breakout_level is not None:
            if latest_price > breakout_level * 1.001:
                resistance_breakout_status = "Breakout"
            elif latest_price >= breakout_level * 0.97:
                resistance_breakout_status = "Near Resistance"
            else:
                resistance_breakout_status = "Below Resistance"
            if df["high"].iloc[-1] > breakout_level and latest_price < breakout_level:
                resistance_breakout_status = "Failed Breakout"
        if nearest_support is not None:
            support_hold_status = "Holding Support" if latest_price >= nearest_support else "Support Broken"

    return {
        "support_20d": support_20d,
        "support_50d": support_50d,
        "support_100d": support_100d,
        "resistance_20d": resistance_20d,
        "resistance_50d": resistance_50d,
        "resistance_100d": resistance_100d,
        "pivot_highs": pivot_highs,
        "pivot_lows": pivot_lows,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "breakout_level": breakout_level,
        "support_level": nearest_support,
        "resistance_breakout_status": resistance_breakout_status,
        "support_hold_status": support_hold_status,
    }


def percent_distance(price: float | None, level: float | None) -> float | None:
    if price is None or level is None or price == 0:
        return None
    return (price - level) / price * 100
