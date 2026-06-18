import pandas as pd


def detect_fair_value_gaps(df: pd.DataFrame, lvn_level: float | None = None) -> dict:
    recent = df.dropna(subset=["high", "low", "open", "close"]).tail(120)
    if len(recent) < 3:
        return _empty()

    atr = recent["ATR_14"].dropna().iloc[-1] if "ATR_14" in recent and not recent["ATR_14"].dropna().empty else None
    avg_body = (recent["close"] - recent["open"]).abs().tail(20).mean()
    bullish = []
    bearish = []
    for idx in range(2, len(recent)):
        a = recent.iloc[idx - 2]
        b = recent.iloc[idx - 1]
        c = recent.iloc[idx]
        body = abs(b["close"] - b["open"])
        displacement = body > avg_body * 1.5 if avg_body and pd.notna(avg_body) else False
        if atr is not None and pd.notna(atr):
            displacement = displacement or (b["high"] - b["low"]) > atr * 1.2
        if a["high"] < c["low"]:
            bullish.append((idx, float(a["high"]), float(c["low"]), displacement))
        if a["low"] > c["high"]:
            bearish.append((idx, float(c["high"]), float(a["low"]), displacement))

    if not bullish:
        result = _empty()
    else:
        idx, low, high, displacement = bullish[-1]
        current_low = float(recent["low"].iloc[idx:].min())
        fill_pct = max(0, min(100, (high - current_low) / (high - low) * 100)) if high > low else 100
        midpoint = (low + high) / 2
        overlaps_lvn = lvn_level is not None and low <= lvn_level <= high
        quality = 20 + (20 if displacement else 0) + (15 if overlaps_lvn else 0) + (10 if fill_pct < 100 else 0)
        result = {
            "bullish_fvg_active": fill_pct < 100,
            "bullish_fvg_low": round(low, 2),
            "bullish_fvg_high": round(high, 2),
            "bullish_fvg_midpoint": round(midpoint, 2),
            "bullish_fvg_age": len(recent) - idx,
            "bullish_fvg_fill_pct": round(fill_pct, 2),
            "bullish_fvg_quality_score": min(100, quality),
            "bullish_fvg_overlaps_lvn": overlaps_lvn,
        }

    if bearish:
        _, low, high, _ = bearish[-1]
        result["bearish_fvg_above"] = high > float(recent["close"].iloc[-1])
        result["bearish_fvg_resistance_zone"] = f"{low:.2f} - {high:.2f}"
    return result


def _empty() -> dict:
    return {
        "bullish_fvg_active": False,
        "bullish_fvg_low": None,
        "bullish_fvg_high": None,
        "bullish_fvg_midpoint": None,
        "bullish_fvg_age": None,
        "bullish_fvg_fill_pct": None,
        "bullish_fvg_quality_score": 0,
        "bullish_fvg_overlaps_lvn": False,
        "bearish_fvg_above": False,
        "bearish_fvg_resistance_zone": None,
    }
