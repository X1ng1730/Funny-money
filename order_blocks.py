import pandas as pd


def detect_order_blocks(df: pd.DataFrame, hvn_level: float | None = None) -> dict:
    recent = df.dropna(subset=["open", "high", "low", "close", "volume"]).tail(100)
    if len(recent) < 25:
        return _empty()

    avg_body = (recent["close"] - recent["open"]).abs().rolling(20).mean()
    candidates = []
    for idx in range(20, len(recent)):
        candle = recent.iloc[idx]
        body = abs(candle["close"] - candle["open"])
        atr = candle.get("ATR_14")
        rvol = candle.get("relative_volume")
        bullish_displacement = (
            candle["close"] > candle["open"]
            and pd.notna(avg_body.iloc[idx])
            and body > avg_body.iloc[idx] * 1.5
            and (pd.isna(atr) or (candle["high"] - candle["low"]) > atr * 1.1)
            and (pd.isna(rvol) or rvol > 1.1)
        )
        if not bullish_displacement:
            continue
        previous = recent.iloc[max(0, idx - 4):idx]
        bearish = previous[previous["close"] < previous["open"]]
        if bearish.empty:
            continue
        block = bearish.iloc[-1]
        zone_low = float(block["low"])
        zone_high = float(max(block["open"], block["high"]))
        candidates.append((idx, zone_low, zone_high, float(rvol) if pd.notna(rvol) else None))

    if not candidates:
        return _empty()

    idx, low, high, rvol = candidates[-1]
    current = float(recent["close"].iloc[-1])
    overlaps_hvn = hvn_level is not None and low <= hvn_level <= high
    tested = bool(recent["low"].iloc[idx + 1:].le(high).any()) if idx + 1 < len(recent) else False
    violated = bool(recent["close"].iloc[idx + 1:].lt(low).any()) if idx + 1 < len(recent) else False
    quality = 20 + (15 if rvol and rvol > 1.2 else 0) + (15 if overlaps_hvn else 0) + (10 if not violated else 0) + (5 if tested else 0)
    status = "violated" if violated else "tested_held" if tested else "untested"
    return {
        "bullish_order_block_active": not violated,
        "bullish_order_block_low": round(low, 2),
        "bullish_order_block_high": round(high, 2),
        "bullish_order_block_midpoint": round((low + high) / 2, 2),
        "bullish_order_block_quality_score": min(100, quality),
        "bullish_order_block_overlaps_hvn": overlaps_hvn,
        "bullish_order_block_overlaps_support": low <= current <= high * 1.03,
        "bullish_order_block_test_status": status,
        "nearest_bullish_order_block_below": round((low + high) / 2, 2) if high < current else None,
        "bearish_order_block_above": False,
        "bearish_order_block_zone": None,
    }


def _empty() -> dict:
    return {
        "bullish_order_block_active": False,
        "bullish_order_block_low": None,
        "bullish_order_block_high": None,
        "bullish_order_block_midpoint": None,
        "bullish_order_block_quality_score": 0,
        "bullish_order_block_overlaps_hvn": False,
        "bullish_order_block_overlaps_support": False,
        "bullish_order_block_test_status": "unavailable",
        "nearest_bullish_order_block_below": None,
        "bearish_order_block_above": False,
        "bearish_order_block_zone": None,
    }
