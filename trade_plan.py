from dataclasses import dataclass

import pandas as pd


@dataclass
class TradePlan:
    entry_type: str
    entry_zone: str
    entry_price_low: float | None
    entry_price_high: float | None
    entry_trigger: str
    stop_price: float | None
    stop_reason: str
    target_1: float | None
    target_2: float | None
    target_3: float | None
    target_reason: str
    risk_per_share: float | None
    reward_to_target_1: float | None
    reward_to_target_2: float | None
    risk_reward_target_1: float | None
    risk_reward_target_2: float | None
    suggested_position_size: str
    invalidation_conditions: list[str]
    entry_zone_source: str
    stop_source: str
    target_source: str
    confluence_components: str
    acceptance_required: str
    invalidation_level: float | None
    invalidation_reason: str
    wait_condition: str
    active_or_conditional: str


def _num(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _round(value: float | None) -> float | None:
    return round(float(value), 2) if value is not None and pd.notna(value) else None


def _rr(entry: float | None, stop: float | None, target: float | None) -> tuple[float | None, float | None, float | None]:
    if entry is None or stop is None or target is None:
        return None, None, None
    risk = abs(entry - stop)
    reward = target - entry
    if risk <= 0:
        return _round(risk), _round(reward), None
    return _round(risk), _round(reward), _round(reward / risk)


def generate_trade_plan(row: pd.Series, strategy_name: str) -> TradePlan:
    price = _num(row, "current_price")
    atr = _num(row, "atr_14")
    ema_8 = _num(row, "ema_8")
    ema_9 = _num(row, "ema_9")
    ema_21 = _num(row, "ema_21")
    support = _num(row, "support_level") or _num(row, "support_20d") or _num(row, "support_50d")
    resistance = _num(row, "nearest_resistance") or _num(row, "resistance_20d") or _num(row, "resistance_50d")
    breakout = _num(row, "breakout_level") or resistance
    low = _num(row, "previous_low")
    high = _num(row, "previous_high")
    high_52w = _num(row, "high_52w")
    fvg_low = _num(row, "bullish_fvg_low")
    fvg_high = _num(row, "bullish_fvg_high")
    ob_low = _num(row, "bullish_order_block_low")
    ob_high = _num(row, "bullish_order_block_high")
    hvn_above = _num(row, "vp_nearest_hvn_above")
    hvn_below = _num(row, "vp_nearest_hvn_below")
    poc = _num(row, "vp_poc")
    value_high = _num(row, "vp_value_area_high")
    swept_level = _num(row, "swept_level")
    vwap = _num(row, "intraday_vwap")

    atr_buffer = (atr or 0) * 0.25
    entry_type = "conditional"
    entry_low = price
    entry_high = price
    trigger = "Wait for price confirmation on the daily chart."
    stop = None
    stop_reason = "Technical invalidation level is unavailable."
    invalidation = ["Close below the relevant daily support level."]
    entry_source = "current structure"
    stop_source = "support"
    target_source = "resistance / R-multiple"
    acceptance_required = "Daily close must confirm acceptance; intraday VWAP is an OHLCV proxy unless intraday data is available."
    wait_condition = "Wait for confirmation rather than treating the plan as an execution signal."

    if strategy_name == "Catalyst Gap / Multi-Month Breakout":
        if row.get("setup_type") == "Extended":
            entry_type = "wait_for_pullback"
            preferred = [value for value in [breakout, fvg_low, ob_low, ema_9, ema_8, vwap] if value is not None]
            entry_low = min(preferred) if preferred else price
            entry_high = max(value for value in [breakout, fvg_high, ob_high, ema_8, price] if value is not None)
            trigger = "Wait for pullback to accepted breakout/FVG/order-block/9 EMA confluence, then daily hold."
            entry_source = "accepted breakout retest / estimated FVG or order block / 8-9 EMA"
        else:
            entry_type = "wait_for_breakout"
            entry_low = breakout or price
            entry_high = price
            trigger = "Conditional only: wait for acceptance above breakout level."
            entry_source = "accepted breakout level"
        stop_base = support or ob_low or fvg_low or breakout or low
        stop = stop_base - atr_buffer if stop_base is not None else None
        stop_reason = "Below breakout/FVG/order-block/support with a small ATR buffer."
        stop_source = "breakout/FVG/order-block/support"
        invalidation = ["Close back below breakout level.", "Gap fails and closes below prior resistance."]
    elif strategy_name == "EMA Pullback Trend Continuation":
        entry_type = "conditional"
        ema_candidates = [value for value in [ema_8, ema_9, ema_21, fvg_low, fvg_high, ob_low, ob_high, hvn_below, vwap] if value is not None]
        entry_low = min(ema_candidates) if ema_candidates else price
        entry_high = max(ema_candidates) if ema_candidates else price
        trigger = "Watch closely: reclaim previous-day low, break/retest previous-day high, close above 9 EMA, or VWAP reclaim."
        entry_source = "EMA + FVG/order-block/HVN/VWAP confluence"
        stop_base = min(value for value in [swept_level, ob_low, fvg_low, support, low, ema_21] if value is not None) if any(value is not None for value in [swept_level, ob_low, fvg_low, support, low, ema_21]) else None
        stop = stop_base - atr_buffer if stop_base is not None else None
        stop_reason = "Below sweep low/order-block/FVG/support or below the 21 EMA zone."
        stop_source = "sweep low / FVG / order block / support"
        invalidation = ["Close below 21 EMA with heavy volume.", "Break below pullback support."]
    elif strategy_name == "Reversal / Reclaim Setup":
        entry_type = "conditional"
        entry_low = min(value for value in [swept_level, ema_8, ema_9, poc, value_high, fvg_low, ob_low, price] if value is not None)
        entry_high = max(value for value in [ema_8, ema_9, poc, value_high, breakout, price] if value is not None)
        trigger = "Conditional only: sweep/reclaim, value-area/POC reclaim, 8/9 EMA reclaim, or 200 SMA acceptance."
        entry_source = "sweep reclaim / 8-9 EMA / POC or value-area reclaim"
        stop_base = swept_level or ob_low or fvg_low or support or low
        stop = stop_base - atr_buffer if stop_base is not None else None
        stop_reason = "Below sweep low, reclaim candle low, estimated order block/FVG, or support."
        stop_source = "sweep low / reclaim support"
        invalidation = ["New lower low after reclaim attempt.", "Close back below 8/9 EMA without support."]
    else:
        entry_type = "conditional"
        stop_base = support or low
        stop = stop_base - atr_buffer if stop_base is not None else None

    entry_mid = None
    if entry_low is not None and entry_high is not None:
        entry_mid = (entry_low + entry_high) / 2
    elif price is not None:
        entry_mid = price

    target_1 = hvn_above or poc if price is not None and poc is not None and price < poc else resistance
    if target_1 is None:
        target_1 = value_high or resistance
    if target_1 is None and entry_mid is not None and stop is not None:
        target_1 = entry_mid + abs(entry_mid - stop) * 2
    target_2 = None
    if entry_mid is not None and stop is not None:
        target_2 = entry_mid + abs(entry_mid - stop) * 3
    if resistance is not None and entry_mid is not None and resistance > entry_mid:
        target_2 = max(target_2 or resistance, resistance)
    if hvn_above is not None and entry_mid is not None and hvn_above > entry_mid:
        target_2 = max(target_2 or hvn_above, hvn_above)
    target_3 = high_52w or target_2
    target_source = "nearest HVN/POC/value-area high/resistance or 2R/3R projection"

    risk, reward_1, rr_1 = _rr(entry_mid, stop, target_1)
    _, reward_2, rr_2 = _rr(entry_mid, stop, target_2)
    if rr_1 is not None and rr_1 < 1.5:
        entry_type = "not_ready"
        invalidation.append("Reward/risk is below 1.5:1.")

    entry_zone = "N/A"
    if entry_low is not None and entry_high is not None:
        entry_zone = f"{entry_low:.2f} - {entry_high:.2f}"

    return TradePlan(
        entry_type=entry_type,
        entry_zone=entry_zone,
        entry_price_low=_round(entry_low),
        entry_price_high=_round(entry_high),
        entry_trigger=trigger,
        stop_price=_round(stop),
        stop_reason=stop_reason,
        target_1=_round(target_1),
        target_2=_round(target_2),
        target_3=_round(target_3),
        target_reason="Prior resistance, 52-week level, or 2R/3R projection when no clean overhead level exists.",
        risk_per_share=risk,
        reward_to_target_1=reward_1,
        reward_to_target_2=reward_2,
        risk_reward_target_1=rr_1,
        risk_reward_target_2=rr_2,
        suggested_position_size="Provide account risk dollars later to calculate shares; do not size from full account or margin.",
        invalidation_conditions=invalidation,
        entry_zone_source=entry_source,
        stop_source=stop_source,
        target_source=target_source,
        confluence_components=str(row.get("confluence_components") or ""),
        acceptance_required=acceptance_required,
        invalidation_level=_round(stop),
        invalidation_reason=stop_reason,
        wait_condition=wait_condition,
        active_or_conditional=entry_type,
    )
