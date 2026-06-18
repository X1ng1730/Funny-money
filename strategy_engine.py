from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd

from trade_plan import TradePlan, generate_trade_plan


@dataclass
class StrategyResult:
    ticker: str
    strategy_name: str
    raw_score: int
    advanced_score: int
    context_score: int
    final_score: float
    label: str
    is_match: bool
    watchlist_priority: str
    reasons: list[str]
    warnings: list[str]
    risk_flags: list[str]
    entry_plan: dict
    stop_plan: dict
    target_plan: dict
    invalidation_conditions: list[str]
    data_quality_score: int
    last_updated: str


StrategyFunction = Callable[[pd.Series], tuple[int, list[str], list[str], list[str], str]]
STRATEGIES: dict[str, StrategyFunction] = {}

STRATEGY_DESCRIPTIONS = {
    "Catalyst Gap / Multi-Month Breakout": "Bullish gap or multi-month resistance breakout with volume, catalyst, and trend confirmation.",
    "EMA Pullback Trend Continuation": "Bullish trend continuation setup where price pulls back into the 8/9/21 EMA zone with support nearby.",
    "Reversal / Reclaim Setup": "Conservative bullish reversal setup requiring seller exhaustion, reclaim behavior, and improving structure.",
}


def register_strategy(name: str):
    def decorator(function: StrategyFunction) -> StrategyFunction:
        STRATEGIES[name] = function
        return function

    return decorator


def _num(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _bool(row: pd.Series, column: str) -> bool:
    return bool(row.get(column)) if row.get(column) is not None and not pd.isna(row.get(column)) else False


def _has_text(row: pd.Series, column: str) -> bool:
    value = row.get(column)
    return value is not None and not pd.isna(value) and str(value).strip() not in {"", "N/A", "None"}


def _earnings_within(row: pd.Series, days: int) -> bool:
    value = row.get("next_earnings_date")
    if value in [None, "", "N/A"] or pd.isna(value):
        return False
    try:
        earnings_date = pd.to_datetime(value).to_pydatetime()
    except Exception:
        return False
    return datetime.now() <= earnings_date <= datetime.now() + timedelta(days=days)


def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


def _label(strategy_name: str, score: float) -> str:
    if strategy_name == "Catalyst Gap / Multi-Month Breakout":
        if score >= 85:
            return "A+ breakout watch"
        if score >= 75:
            return "Strong breakout watch"
        if score >= 65:
            return "Developing breakout"
        if score >= 50:
            return "Watchlist only / not ready"
        return "No valid breakout setup"
    if strategy_name == "EMA Pullback Trend Continuation":
        if score >= 85:
            return "A+ pullback continuation"
        if score >= 75:
            return "Strong pullback continuation"
        if score >= 65:
            return "Developing pullback setup"
        if score >= 50:
            return "Watch only / wait for confirmation"
        return "No valid pullback setup"
    if score >= 85:
        return "A+ reclaim/reversal"
    if score >= 75:
        return "Strong reclaim setup"
    if score >= 65:
        return "Developing reclaim"
    if score >= 50:
        return "Watch only / needs confirmation"
    return "No valid reversal setup"


def _priority(score: float) -> str:
    if score >= 85:
        return "High"
    if score >= 70:
        return "Medium"
    if score >= 55:
        return "Low"
    return "Avoid for now"


def _base_filters(row: pd.Series) -> tuple[list[str], list[str], bool]:
    warnings: list[str] = []
    risk_flags: list[str] = []
    fatal = False
    price = _num(row, "current_price")
    market_cap = _num(row, "market_cap")
    avg_volume = _num(row, "average_volume_20d")
    dollar_volume = _num(row, "dollar_volume")

    if row.get("data_status") != "OK":
        warnings.append("Data is missing or unavailable.")
        risk_flags.append("Data Missing")
        fatal = True
    if price is not None and price <= 5:
        warnings.append("Price is below the conservative $5 default filter.")
        risk_flags.append("Low Price")
    if market_cap is not None and market_cap < 1_000_000_000:
        warnings.append("Market cap is below $1B.")
        risk_flags.append("Small / speculative")
    if market_cap is None:
        warnings.append("Market cap is missing.")
        risk_flags.append("Missing Market Cap")
    if (avg_volume is not None and avg_volume < 500_000) or (dollar_volume is not None and dollar_volume < 25_000_000):
        warnings.append("Liquidity is below the default threshold.")
        risk_flags.append("Low Liquidity")
    return warnings, risk_flags, fatal


def _apply_penalties(row: pd.Series, score: int, warnings: list[str], risk_flags: list[str]) -> int:
    price = _num(row, "current_price")
    ema_8 = _num(row, "ema_8")
    ema_9 = _num(row, "ema_9")
    ema_21 = _num(row, "ema_21")
    sma_200 = _num(row, "sma_200")
    rsi = _num(row, "rsi_14")
    atr_pct = _num(row, "atr_pct")

    if price is not None and ema_8 is not None and price > ema_8 * 1.08:
        score -= 15
        warnings.append("Price is extended more than 8% above 8 EMA.")
        risk_flags.append("Extended Above 8/9 EMA")
    if price is not None and ema_21 is not None and price > ema_21 * 1.10:
        score -= 10
        warnings.append("Price is extended more than 10% above 21 EMA.")
        risk_flags.append("Extended Above 8/9 EMA")
    if rsi is not None and rsi > 75:
        score -= 10
        warnings.append("RSI is overbought above 75.")
        risk_flags.append("RSI Overbought")
    if _earnings_within(row, 2):
        score -= 10
        warnings.append("Earnings are within 2 calendar days.")
        risk_flags.append("Earnings Very Soon")
    elif _earnings_within(row, 7):
        risk_flags.append("Earnings Soon")
    if atr_pct is not None and atr_pct > 7:
        warnings.append("ATR% is very high.")
        risk_flags.append("Very High ATR")
    elif atr_pct is not None and atr_pct > 4:
        risk_flags.append("High ATR")
    if price is not None and sma_200 is not None and price < sma_200 and not _bool(row, "crossed_above_200sma_10d"):
        score -= 15
        warnings.append("Price is below 200 SMA without a recent reclaim.")
        risk_flags.append("Below 200 SMA")
    if ema_9 is None:
        risk_flags.append("Data Quality Warning")
    return score


def higher_timeframe_context(row: pd.Series, trade_plan: TradePlan | None = None) -> tuple[int, list[str], list[str]]:
    score = 50
    reasons: list[str] = []
    warnings: list[str] = []
    trend = row.get("trend_status")
    resistance_distance = _num(row, "distance_to_nearest_resistance_pct")
    support_distance = _num(row, "distance_to_nearest_support_pct")
    atr_pct = _num(row, "atr_pct")
    rr = trade_plan.risk_reward_target_1 if trade_plan else None

    if trend in {"Strong Uptrend", "Uptrend"}:
        score += 20
        reasons.append("Daily structure is aligned bullish.")
    elif trend in {"Pullback in Uptrend", "Reclaiming Trend"}:
        score += 12
        reasons.append("Daily structure is constructive but still needs confirmation.")
    elif trend in {"Breakdown Risk", "Long-Term Weak"}:
        score -= 20
        warnings.append("Higher-timeframe context is weak or bearish.")

    if row.get("market_regime") == "Risk-On":
        score += 5
        reasons.append("SPY/QQQ market regime is risk-on.")
    elif row.get("market_regime") == "Risk-Off":
        score -= 8
        warnings.append("SPY/QQQ market regime is risk-off.")

    if resistance_distance is not None and resistance_distance >= 5:
        score += 10
        reasons.append("Nearest resistance leaves at least 5% room.")
    elif resistance_distance is not None and abs(resistance_distance) <= 2:
        score -= 15
        warnings.append("Price is directly under major resistance.")

    if support_distance is not None and support_distance <= 8:
        score += 5
        reasons.append("Support is close enough to define risk.")
    if rr is not None and rr >= 2:
        score += 10
        reasons.append("Trade plan offers at least 2:1 reward/risk.")
    elif rr is not None and rr < 1.5:
        score -= 10
        warnings.append("Reward/risk is below 1.5:1.")
    if atr_pct is not None and atr_pct <= 7:
        score += 5
    if row.get("weekly_acceptance_status") == "above value":
        score += 10
        reasons.append("Weekly volume-profile proxy shows price above value.")
    if _num(row, "weekly_vp_poc") is not None and _num(row, "current_price") is not None and _num(row, "current_price") > _num(row, "weekly_vp_poc"):
        score += 10
        reasons.append("Price is above weekly POC proxy.")
    weekly_room = _num(row, "weekly_resistance_overhead_pct")
    if weekly_room is not None and weekly_room >= 5:
        score += 10
        reasons.append("Weekly HVN/resistance proxy leaves room above.")
    elif weekly_room is not None and abs(weekly_room) <= 2:
        score -= 15
        warnings.append("Price is directly under weekly HVN/resistance proxy.")
    if row.get("weekly_acceptance_status") == "below value":
        score -= 15
        warnings.append("Weekly volume-profile proxy shows price below value.")
    return _clamp(score), reasons, warnings


@register_strategy("Catalyst Gap / Multi-Month Breakout")
def catalyst_gap_breakout(row: pd.Series) -> tuple[int, list[str], list[str], list[str], str]:
    warnings, risk_flags, fatal = _base_filters(row)
    if fatal:
        return 0, [], warnings, risk_flags, "No bullish setup"
    score = 0
    reasons: list[str] = []
    price = _num(row, "current_price")
    ema_8 = _num(row, "ema_8")
    ema_21 = _num(row, "ema_21")
    sma_200 = _num(row, "sma_200")
    rvol = _num(row, "relative_volume")
    gap = _num(row, "gap_pct")
    one_day = _num(row, "return_1d_pct")
    dollar_volume = _num(row, "dollar_volume")

    if price is not None and ema_8 is not None and price > ema_8:
        score += 10; reasons.append("Close is above 8 EMA.")
    if price is not None and ema_21 is not None and price > ema_21:
        score += 5; reasons.append("Close is above 21 EMA.")
    if price is not None and sma_200 is not None and (price > sma_200 or _bool(row, "crossed_above_200sma_10d")):
        score += 10; reasons.append("Price is above or recently reclaimed the 200 SMA.")

    if row.get("resistance_breakout_status") == "Breakout":
        score += 20; reasons.append("Close broke above multi-session resistance.")
    elif row.get("resistance_breakout_status") == "Near Resistance":
        score += 10; reasons.append("Price is within breakout range of resistance.")
    if row.get("support_hold_status") == "Holding Support":
        score += 5; reasons.append("Price is holding above support/breakout structure.")
    if (one_day is not None and one_day > 3) or (gap is not None and gap > 2.5):
        score += 5; reasons.append("Daily move or gap is meaningful.")

    if rvol is not None and rvol > 1.2:
        score += 8; reasons.append("RVOL is above normal.")
    if rvol is not None and rvol > 2:
        score += 7; reasons.append("RVOL is unusually strong.")
    if dollar_volume is not None and dollar_volume > 25_000_000:
        score += 5; reasons.append("Dollar volume clears liquidity threshold.")

    if _has_text(row, "manual_catalyst"):
        score += 8; reasons.append("Manual catalyst is present.")
    if _has_text(row, "latest_headline"):
        score += 5; reasons.append("Latest headline is available.")
    if any(word in str(row.get("latest_headline", "")).lower() for word in ["earnings", "guidance", "upgrade", "contract", "index", "ai"]):
        score += 2; reasons.append("Headline contains catalyst keywords.")

    if (_num(row, "relative_strength_spy_5d") or 0) > 0 or (_num(row, "relative_strength_qqq_5d") or 0) > 0:
        score += 5; reasons.append("5D relative strength is positive.")
    if (_num(row, "relative_strength_spy_1m") or 0) > 0 or (_num(row, "relative_strength_qqq_1m") or 0) > 0:
        score += 5; reasons.append("1M relative strength is positive.")
    if row.get("resistance_breakout_status") == "Failed Breakout":
        score -= 15; warnings.append("Breakout or gap failed."); risk_flags.append("Gap Failed")
    if row.get("accepted_above_breakout_level"):
        score += 10; reasons.append("Breakout has daily acceptance above the key level.")
    if row.get("price_above_vwap"):
        score += 8; reasons.append("Price is above VWAP proxy.")
    if row.get("bullish_fvg_active"):
        score += 8; reasons.append("Estimated bullish FVG remains active.")
    if row.get("vp_nearest_lvn_above") is not None and row.get("vp_nearest_hvn_above") is not None:
        score += 8; reasons.append("Volume-profile proxy shows LVN/HVN path above.")
    if row.get("rejected_at_resistance") or row.get("breakout_sweep_failure_flag"):
        score -= 15; warnings.append("Breakout acceptance failed or swept above resistance."); risk_flags.append("Breakout Rejected")
    return _clamp(_apply_penalties(row, score, warnings, risk_flags)), reasons, warnings, risk_flags, "Catalyst breakout"


@register_strategy("EMA Pullback Trend Continuation")
def ema_pullback_continuation(row: pd.Series) -> tuple[int, list[str], list[str], list[str], str]:
    warnings, risk_flags, fatal = _base_filters(row)
    if fatal:
        return 0, [], warnings, risk_flags, "No bullish setup"
    score = 0
    reasons: list[str] = []
    price = _num(row, "current_price")
    ema_8 = _num(row, "ema_8")
    ema_9 = _num(row, "ema_9")
    ema_21 = _num(row, "ema_21")
    sma_50 = _num(row, "sma_50")
    sma_200 = _num(row, "sma_200")
    rvol = _num(row, "relative_volume")

    if price is not None and sma_200 is not None and price > sma_200:
        score += 10; reasons.append("Close is above 200 SMA.")
    if price is not None and sma_50 is not None and price > sma_50:
        score += 5; reasons.append("Close is above 50 SMA.")
    if ema_21 is not None and ((ema_8 is not None and ema_8 > ema_21) or (ema_9 is not None and ema_9 > ema_21)):
        score += 5; reasons.append("8/9 EMA is above 21 EMA.")
    if (_num(row, "return_1m_pct") or 0) > 0:
        score += 5; reasons.append("1M trend is positive.")

    if price is not None and ema_8 is not None and abs(price - ema_8) / price <= 0.025:
        score += 10; reasons.append("Close is near 8 EMA pullback zone.")
    if price is not None and ema_9 is not None and abs(price - ema_9) / price <= 0.025:
        score += 10; reasons.append("Close is near 9 EMA pullback zone.")
    if price is not None and ema_21 is not None and abs(price - ema_21) / price <= 0.035:
        score += 10; reasons.append("Close is near 21 EMA pullback zone.")
    if row.get("support_hold_status") == "Holding Support":
        score += 5; reasons.append("Nearby support is holding.")

    previous_low = _num(row, "previous_low")
    previous_high = _num(row, "previous_high")
    if price is not None and previous_low is not None and price > previous_low:
        score += 8; reasons.append("Price is holding/reclaiming previous-day low area.")
    if price is not None and previous_high is not None and price > previous_high:
        score += 8; reasons.append("Price broke above previous-day high.")
    if price is not None and _num(row, "high_20d") is not None and price > (_num(row, "high_20d") or 0) * 0.97:
        score += 4; reasons.append("Close is constructive near the recent upper range.")

    if rvol is not None and rvol > 1.0:
        score += 5; reasons.append("Volume is at least normal.")
    if rvol is not None and rvol > 1.2:
        score += 5; reasons.append("Bounce-day RVOL is above normal.")
    if (_num(row, "relative_strength_spy_1m") or 0) > 0 or (_num(row, "relative_strength_qqq_1m") or 0) > 0:
        score += 5; reasons.append("Relative strength is positive.")
    if row.get("bullish_fvg_active") and row.get("confluence_score", 0) >= 20:
        score += 10; reasons.append("EMA pullback overlaps estimated FVG/confluence zone.")
    if row.get("bullish_order_block_active") and row.get("bullish_order_block_overlaps_hvn"):
        score += 10; reasons.append("Estimated bullish order block overlaps HVN proxy.")
    if row.get("bullish_liquidity_sweep_detected"):
        score += 10; reasons.append("Bullish liquidity sweep/reclaim proxy detected.")
    if row.get("price_above_vwap") or row.get("vwap_reclaim_detected"):
        score += 8; reasons.append("VWAP proxy is reclaimed or holding.")
    if row.get("vwap_rejection_detected"):
        score -= 15; warnings.append("Price rejected at VWAP proxy."); risk_flags.append("VWAP Rejection")
    if price is not None and sma_50 is not None and price < sma_50:
        score -= 15; warnings.append("Close is below 50 SMA."); risk_flags.append("Below 50 SMA")
    return _clamp(_apply_penalties(row, score, warnings, risk_flags)), reasons, warnings, risk_flags, "EMA pullback"


@register_strategy("Reversal / Reclaim Setup")
def reversal_reclaim(row: pd.Series) -> tuple[int, list[str], list[str], list[str], str]:
    warnings, risk_flags, fatal = _base_filters(row)
    if fatal:
        return 0, [], warnings, risk_flags, "No bullish setup"
    score = 0
    reasons: list[str] = []
    price = _num(row, "current_price")
    ema_8 = _num(row, "ema_8")
    ema_9 = _num(row, "ema_9")
    sma_50 = _num(row, "sma_50")
    sma_200 = _num(row, "sma_200")
    high_3m = _num(row, "high_100d")
    low_20d = _num(row, "low_20d")
    rsi = _num(row, "rsi_14")

    if price is not None and high_3m is not None and high_3m > 0 and (high_3m - price) / high_3m > 0.15:
        score += 8; reasons.append("Stock has had a meaningful drawdown from recent highs.")
    if rsi is not None and rsi < 40:
        score += 5; reasons.append("RSI recently reflects washed-out conditions.")
    if _num(row, "relative_volume") is not None and (_num(row, "relative_volume") or 0) > 1.2:
        score += 7; reasons.append("Volume is active during reclaim attempt.")

    if price is not None and ((ema_8 is not None and price > ema_8) or (ema_9 is not None and price > ema_9)):
        score += 10; reasons.append("Close reclaimed 8/9 EMA.")
    if price is not None and low_20d is not None and price > low_20d * 1.03:
        score += 10; reasons.append("Price is off recent lows, suggesting a bottom attempt.")
    if row.get("resistance_breakout_status") == "Breakout" or _bool(row, "crossed_above_21ema_5d"):
        score += 10; reasons.append("Short-term downtrend proxy or 21 EMA has been reclaimed.")

    if price is not None and sma_50 is not None and (price > sma_50 or _bool(row, "crossed_above_21ema_5d")):
        score += 10; reasons.append("Price is near/above intermediate trend reclaim.")
    if price is not None and sma_200 is not None and (price > sma_200 or abs(price - sma_200) / price <= 0.03 or _bool(row, "crossed_above_200sma_10d")):
        score += 10; reasons.append("Price is above or close to reclaiming the 200 SMA.")

    if _has_text(row, "manual_catalyst") or _has_text(row, "latest_headline"):
        score += 5; reasons.append("Catalyst/news exists for the reclaim attempt.")
    if row.get("bullish_liquidity_sweep_detected"):
        score += 12; reasons.append("Bullish liquidity sweep below support and reclaim detected.")
    if row.get("vp_current_location") in {"inside value", "above value"}:
        score += 10; reasons.append("Price reclaimed value area/POC proxy context.")
    if row.get("vwap_reclaim_detected") or row.get("price_above_vwap"):
        score += 10; reasons.append("VWAP proxy reclaimed after weakness.")
    if row.get("bullish_fvg_active"):
        score += 10; reasons.append("Estimated bullish FVG formed after displacement.")
    if row.get("bullish_order_block_active") and row.get("bullish_order_block_overlaps_hvn"):
        score += 10; reasons.append("Estimated order block with HVN overlap supports reclaim.")
    if row.get("vp_current_location") == "below value":
        score -= 20; warnings.append("Price is still accepting below value area proxy."); risk_flags.append("Below Value Area")
    if price is not None and ema_8 is not None and price < ema_8:
        score -= 20; warnings.append("Close is still below 8 EMA."); risk_flags.append("Not Ready Yet")
    if row.get("trend_status") == "Long-Term Weak" and not _bool(row, "crossed_above_200sma_10d"):
        score -= 10; warnings.append("Still far from major trend reclaim."); risk_flags.append("Potential Reversal Only")
    return _clamp(_apply_penalties(row, score, warnings, risk_flags)), reasons, warnings, risk_flags, "Reversal/reclaim"


def evaluate_strategy(row: pd.Series, strategy_name: str) -> StrategyResult:
    raw_score, reasons, warnings, risk_flags, _ = STRATEGIES[strategy_name](row)
    plan = generate_trade_plan(row, strategy_name)
    context_score, context_reasons, context_warnings = higher_timeframe_context(row, plan)
    advanced_score = int(_num(row, "advanced_technical_score") or 0)
    final_score = raw_score * 0.55 + advanced_score * 0.25 + context_score * 0.20

    data_quality = int(_num(row, "data_quality_score") or (0 if row.get("data_status") != "OK" else 70))
    if context_score < 45:
        final_score = min(final_score, 65)
    if "Low Liquidity" in risk_flags:
        final_score = min(final_score, 70)
    if strategy_name == "Catalyst Gap / Multi-Month Breakout" and row.get("breakout_acceptance_status") == "failed":
        final_score = min(final_score, 65)
    if strategy_name != "Reversal / Reclaim Setup" and _num(row, "current_price") is not None and _num(row, "sma_200") is not None and _num(row, "current_price") < _num(row, "sma_200"):
        final_score = min(final_score, 65)
    if strategy_name == "EMA Pullback Trend Continuation" and _num(row, "current_price") is not None and _num(row, "ema_21") is not None and _num(row, "current_price") < _num(row, "ema_21"):
        final_score = min(final_score, 60)
    if strategy_name == "Reversal / Reclaim Setup" and not row.get("bullish_liquidity_sweep_detected") and not row.get("vwap_reclaim_detected") and not row.get("crossed_above_8ema_3d"):
        final_score = min(final_score, 65)
    if data_quality < 60:
        final_score = min(final_score, 60)
        risk_flags.append("Data Quality Warning")
    if row.get("market_regime") == "Risk-On":
        final_score = min(100, final_score + 3)
    elif row.get("market_regime") == "Risk-Off":
        final_score = max(0, final_score - 5)

    final_score = round(final_score, 1)
    all_reasons = reasons + context_reasons
    all_warnings = warnings + context_warnings
    risk_flags = sorted(set([flag for flag in risk_flags + str(row.get("risk_flags", "")).split(", ") if flag]))

    return StrategyResult(
        ticker=str(row.get("ticker", "")),
        strategy_name=strategy_name,
        raw_score=raw_score,
        advanced_score=advanced_score,
        context_score=context_score,
        final_score=final_score,
        label=_label(strategy_name, final_score),
        is_match=final_score >= 65,
        watchlist_priority=_priority(final_score),
        reasons=all_reasons or ["No bullish technical setup confirmed yet."],
        warnings=all_warnings,
        risk_flags=risk_flags,
        entry_plan={
            "entry_type": plan.entry_type,
            "entry_zone": plan.entry_zone,
            "entry_price_low": plan.entry_price_low,
            "entry_price_high": plan.entry_price_high,
            "entry_trigger": plan.entry_trigger,
            "entry_zone_source": plan.entry_zone_source,
            "acceptance_required": plan.acceptance_required,
            "wait_condition": plan.wait_condition,
        },
        stop_plan={
            "stop_price": plan.stop_price,
            "stop_reason": plan.stop_reason,
            "stop_source": plan.stop_source,
            "invalidation_level": plan.invalidation_level,
            "invalidation_reason": plan.invalidation_reason,
        },
        target_plan={
            "target_1": plan.target_1,
            "target_2": plan.target_2,
            "target_3": plan.target_3,
            "target_reason": plan.target_reason,
            "target_source": plan.target_source,
            "risk_per_share": plan.risk_per_share,
            "reward_to_target_1": plan.reward_to_target_1,
            "reward_to_target_2": plan.reward_to_target_2,
            "risk_reward_target_1": plan.risk_reward_target_1,
            "risk_reward_target_2": plan.risk_reward_target_2,
        },
        invalidation_conditions=plan.invalidation_conditions,
        data_quality_score=data_quality,
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def run_strategies(row: pd.Series, strategy_name: str | None = None) -> list[StrategyResult]:
    strategies = STRATEGIES
    if strategy_name and strategy_name != "All Strategies":
        strategies = {strategy_name: STRATEGIES[strategy_name]} if strategy_name in STRATEGIES else {}
    return [evaluate_strategy(row, name) for name in strategies]


def result_to_flat_row(row: pd.Series, result: StrategyResult) -> dict:
    combined = row.to_dict()
    combined.update(
        {
            "best_strategy": result.strategy_name,
            "strategy": result.strategy_name,
            "strategy_name": result.strategy_name,
            "raw_strategy_score": result.raw_score,
            "advanced_technical_score": result.advanced_score,
            "advanced_score": result.advanced_score,
            "context_score": result.context_score,
            "final_strategy_score": result.final_score,
            "strategy_score": result.final_score,
            "match_label": result.label,
            "strategy_match": result.is_match,
            "watchlist_priority": result.watchlist_priority,
            "reasons": " ".join(result.reasons),
            "warnings": " ".join(result.warnings),
            "risk_flags": ", ".join(result.risk_flags),
            "entry_type": result.entry_plan.get("entry_type"),
            "entry_zone": result.entry_plan.get("entry_zone"),
            "entry_trigger": result.entry_plan.get("entry_trigger"),
            "entry_zone_source": result.entry_plan.get("entry_zone_source"),
            "acceptance_required": result.entry_plan.get("acceptance_required"),
            "wait_condition": result.entry_plan.get("wait_condition"),
            "stop_price": result.stop_plan.get("stop_price"),
            "stop_reason": result.stop_plan.get("stop_reason"),
            "stop_source": result.stop_plan.get("stop_source"),
            "target_1": result.target_plan.get("target_1"),
            "target_2": result.target_plan.get("target_2"),
            "target_3": result.target_plan.get("target_3"),
            "target_source": result.target_plan.get("target_source"),
            "risk_per_share": result.target_plan.get("risk_per_share"),
            "risk_reward_target_1": result.target_plan.get("risk_reward_target_1"),
            "risk_reward_target_2": result.target_plan.get("risk_reward_target_2"),
            "invalidation_conditions": " ".join(result.invalidation_conditions),
            "data_quality_score": result.data_quality_score,
            "last_updated": result.last_updated,
            "strategy_result": asdict(result),
        }
    )
    return combined


def build_strategy_results(dashboard_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in dashboard_df.iterrows():
        for result in run_strategies(row):
            rows.append(result_to_flat_row(row, result))
    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values("final_strategy_score", ascending=False).reset_index(drop=True)
    return result_df


def best_strategy_per_ticker(strategy_df: pd.DataFrame) -> pd.DataFrame:
    if strategy_df.empty:
        return strategy_df
    return strategy_df.sort_values("final_strategy_score", ascending=False).drop_duplicates("ticker", keep="first")
