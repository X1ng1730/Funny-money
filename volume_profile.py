import pandas as pd


def _empty_profile() -> dict:
    return {
        "vp_poc": None,
        "vp_value_area_high": None,
        "vp_value_area_low": None,
        "vp_nearest_hvn_above": None,
        "vp_nearest_hvn_below": None,
        "vp_nearest_lvn_above": None,
        "vp_nearest_lvn_below": None,
        "vp_current_location": "profile unavailable",
        "vp_value_area_width_pct": None,
        "vp_acceptance_score": 0,
        "vp_rejection_score": 0,
    }


def calculate_volume_profile(df: pd.DataFrame, lookback: int = 60, bins: int = 50) -> dict:
    recent = df.tail(lookback).dropna(subset=["high", "low", "close", "volume"])
    if recent.empty or recent["high"].max() <= recent["low"].min():
        return _empty_profile()

    low = float(recent["low"].min())
    high = float(recent["high"].max())
    bin_edges = pd.interval_range(start=low, end=high, periods=bins)
    volumes = pd.Series(0.0, index=bin_edges)

    # OHLCV proxy: distribute candle volume evenly across touched price bins.
    for _, candle in recent.iterrows():
        touched = [interval for interval in bin_edges if interval.left <= candle["high"] and interval.right >= candle["low"]]
        if not touched:
            continue
        allocation = float(candle["volume"]) / len(touched)
        for interval in touched:
            volumes.loc[interval] += allocation

    if volumes.sum() <= 0:
        return _empty_profile()

    poc_interval = volumes.idxmax()
    poc = (poc_interval.left + poc_interval.right) / 2
    hvn_threshold = volumes.quantile(0.75)
    lvn_threshold = volumes.quantile(0.25)
    hvns = sorted([(i.left + i.right) / 2 for i, v in volumes.items() if v >= hvn_threshold])
    lvns = sorted([(i.left + i.right) / 2 for i, v in volumes.items() if v <= lvn_threshold])

    ordered = volumes.sort_values(ascending=False)
    cumulative = ordered.cumsum()
    value_bins = ordered[cumulative <= volumes.sum() * 0.70]
    if value_bins.empty:
        value_bins = ordered.head(1)
    value_area_low = min(interval.left for interval in value_bins.index)
    value_area_high = max(interval.right for interval in value_bins.index)
    current = float(recent["close"].iloc[-1])

    def nearest(levels: list[float], above: bool) -> float | None:
        candidates = [level for level in levels if level > current] if above else [level for level in levels if level < current]
        if not candidates:
            return None
        return min(candidates) if above else max(candidates)

    if current > value_area_high:
        location = "above value"
    elif current < value_area_low:
        location = "below value"
    else:
        location = "inside value"

    acceptance_score = 0
    rejection_score = 0
    if current > value_area_high:
        acceptance_score += 45
    elif location == "inside value":
        acceptance_score += 25
    if recent["low"].iloc[-1] < value_area_low and current > value_area_low:
        rejection_score += 60
    if recent["high"].iloc[-1] > value_area_high and current < value_area_high:
        rejection_score += 60

    return {
        "vp_poc": round(poc, 2),
        "vp_value_area_high": round(value_area_high, 2),
        "vp_value_area_low": round(value_area_low, 2),
        "vp_nearest_hvn_above": nearest(hvns, True),
        "vp_nearest_hvn_below": nearest(hvns, False),
        "vp_nearest_lvn_above": nearest(lvns, True),
        "vp_nearest_lvn_below": nearest(lvns, False),
        "vp_current_location": location,
        "vp_value_area_width_pct": round((value_area_high - value_area_low) / current * 100, 2) if current else None,
        "vp_acceptance_score": min(100, acceptance_score),
        "vp_rejection_score": min(100, rejection_score),
        "volume_by_price": pd.DataFrame(
            {
                "price_low": [i.left for i in volumes.index],
                "price_high": [i.right for i in volumes.index],
                "estimated_volume": volumes.values,
            }
        ),
    }
