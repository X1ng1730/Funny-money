import pandas as pd


def _num(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def evaluate_acceptance_rejection(row: pd.Series, level: float | None, level_name: str = "key level") -> dict:
    close = _num(row.get("close"))
    high = _num(row.get("high"))
    low = _num(row.get("low"))
    open_price = _num(row.get("open"))
    atr = _num(row.get("ATR_14"))
    rvol = _num(row.get("relative_volume"))
    if None in [close, high, low, level] or high == low:
        return {
            "accepted_above_breakout_level": False,
            "rejected_at_resistance": False,
            "bullish_rejection_at_support": False,
            "acceptance_score": 0,
            "rejection_score": 0,
            "breakout_acceptance_status": "unconfirmed",
            "key_level_status": f"{level_name}: unavailable",
        }

    buffer = max(level * 0.002, (atr or 0) * 0.10)
    candle_range = high - low
    close_position = (close - low) / candle_range
    upper_wick = (high - max(open_price or close, close)) / candle_range
    lower_wick = (min(open_price or close, close) - low) / candle_range

    acceptance_score = 0
    if close > level:
        acceptance_score += 25
    if close > level + buffer:
        acceptance_score += 15
    if rvol is not None and rvol > 1.2:
        acceptance_score += 15
    if close_position >= 0.50:
        acceptance_score += 15
    if low <= level <= close:
        acceptance_score += 15
    if close > level and close_position >= 0.60:
        acceptance_score += 15

    rejection_score = 0
    bullish_rejection = low < level and close > level
    rejected_at_resistance = high > level and close < level
    if bullish_rejection or rejected_at_resistance:
        rejection_score += 25
    if bullish_rejection:
        rejection_score += 20
    if rejected_at_resistance:
        rejection_score += 20
    if max(lower_wick, upper_wick) > 0.40:
        rejection_score += 15
    if rvol is not None and rvol > 1.0:
        rejection_score += 15
    if bullish_rejection and close_position >= 0.50:
        rejection_score += 15
    if rejected_at_resistance and close_position <= 0.50:
        rejection_score += 15

    if close > level + buffer and acceptance_score >= 55:
        status = "accepted"
    elif rejected_at_resistance:
        status = "failed"
    elif close > level:
        status = "unconfirmed"
    else:
        status = "rejected"

    return {
        "accepted_above_breakout_level": status == "accepted",
        "rejected_at_resistance": rejected_at_resistance,
        "bullish_rejection_at_support": bullish_rejection,
        "acceptance_score": min(100, acceptance_score),
        "rejection_score": min(100, rejection_score),
        "breakout_acceptance_status": status,
        "key_level_status": f"{level_name}: {status}",
    }
