import pandas as pd


def _num(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _near(a: float | None, b: float | None, price: float | None, tolerance_pct: float = 2.0) -> bool:
    if a is None or b is None or price is None or price == 0:
        return False
    return abs(a - b) / price * 100 <= tolerance_pct


def find_confluence_zones(row: pd.Series) -> list[dict]:
    price = _num(row, "current_price")
    if price is None:
        return []
    zones = []
    candidate_levels = {
        "EMA 8": _num(row, "ema_8"),
        "EMA 9": _num(row, "ema_9"),
        "EMA 21": _num(row, "ema_21"),
        "support": _num(row, "support_level"),
        "breakout": _num(row, "breakout_level"),
        "FVG midpoint": _num(row, "bullish_fvg_midpoint"),
        "order block": _num(row, "bullish_order_block_midpoint"),
        "HVN": _num(row, "vp_nearest_hvn_below"),
        "LVN": _num(row, "vp_nearest_lvn_above"),
        "VWAP proxy": _num(row, "intraday_vwap"),
    }
    for name, level in candidate_levels.items():
        if level is None:
            continue
        components = [other for other, other_level in candidate_levels.items() if other != name and _near(level, other_level, price)]
        if not components:
            continue
        components.append(name)
        score = min(100, len(set(components)) * 10)
        if row.get("bullish_liquidity_sweep_detected"):
            score += 10
            components.append("liquidity sweep/reclaim")
        if (row.get("acceptance_score") or 0) >= 60:
            score += 10
            components.append("acceptance")
        zones.append(
            {
                "zone_low": round(level * 0.995, 2),
                "zone_high": round(level * 1.005, 2),
                "zone_mid": round(level, 2),
                "confluence_score": min(100, score),
                "components": sorted(set(components)),
                "zone_type": "support" if level <= price else "resistance",
                "suggested_use": "entry" if level <= price else "target",
            }
        )
    return sorted(zones, key=lambda item: item["confluence_score"], reverse=True)


def confluence_summary(row: pd.Series) -> dict:
    zones = find_confluence_zones(row)
    best = zones[0] if zones else None
    return {
        "confluence_score": best["confluence_score"] if best else 0,
        "confluence_zone": f"{best['zone_low']} - {best['zone_high']}" if best else None,
        "confluence_components": ", ".join(best["components"]) if best else "",
        "confluence_zone_type": best["zone_type"] if best else None,
        "confluence_suggested_use": best["suggested_use"] if best else None,
        "confluence_zones": zones,
    }
