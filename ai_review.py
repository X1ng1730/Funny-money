from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
import json

import pandas as pd

from ai_prompt_templates import AI_REVIEW_PROMPT_VERSION, build_candidate_review_prompt
from ai_validation import validate_ai_review
from ollama_client import call_ollama_json, check_ollama_available

AI_LOG_DIR = Path("data") / "ai_reviews"
AI_LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AIReviewSettings:
    enabled: bool = False
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    timeout: int = 30
    candidate_limit: int = 25
    min_deterministic_score: float = 60
    allow_score_influence: bool = True
    ai_weight: float = 0.15
    allow_caps: bool = True
    allow_positive_adjustment: bool = True
    show_raw_json: bool = False
    force_rerun: bool = False


@dataclass
class AIReviewResult:
    ticker: str
    strategy_name: str
    ai_available: bool
    ai_valid: bool
    setup_quality: str
    watchlist_action: str
    ai_review_score: int
    component_scores: dict[str, int]
    best_strategy_fit: str
    strategy_fit_comment: str
    trade_maturity: str
    entry_quality: str
    stop_quality: str
    target_quality: str
    main_reason: str
    confirmation_needed: list[str]
    entry_comment: str
    stop_comment: str
    target_comment: str
    catalyst_interpretation: str
    catalyst_comment: str
    main_risks: list[str]
    downgrade_reasons: list[str]
    missing_data_notes: list[str]
    confidence: str
    ai_score_adjustment: int
    raw_response: dict[str, Any] | str
    error_message: str


def _safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def build_candidate_packet(row: pd.Series, competing: pd.DataFrame | None = None) -> dict[str, Any]:
    support_levels = {key: _safe(row.get(key)) for key in ["support_20d", "support_50d", "support_100d", "support_level"]}
    resistance_levels = {key: _safe(row.get(key)) for key in ["resistance_20d", "resistance_50d", "resistance_100d", "nearest_resistance"]}
    missing = [key for key, value in row.items() if _safe(value) is None]
    competing_scores = []
    if competing is not None and not competing.empty:
        for _, comp in competing.iterrows():
            competing_scores.append(
                {
                    "strategy": comp.get("strategy"),
                    "deterministic_final_score": comp.get("final_strategy_score"),
                    "base_strategy_score": comp.get("raw_strategy_score"),
                    "advanced_technical_score": comp.get("advanced_technical_score"),
                    "context_score": comp.get("context_score"),
                }
            )
    return {
        "ticker": row.get("ticker"),
        "category": row.get("category"),
        "quick_thesis": row.get("quick_thesis"),
        "macro_tag": row.get("macro_tag"),
        "manual_catalyst": row.get("manual_catalyst"),
        "latest_headline": row.get("latest_headline"),
        "earnings_date": row.get("next_earnings_date"),
        "current_price": row.get("current_price"),
        "price_change_1d": row.get("return_1d_pct"),
        "price_change_5d": row.get("return_5d_pct"),
        "price_change_1m": row.get("return_1m_pct"),
        "volume": row.get("volume"),
        "relative_volume": row.get("relative_volume"),
        "dollar_volume": row.get("dollar_volume"),
        "market_cap": row.get("market_cap"),
        "liquidity_rating": "low" if "Low Liquidity" in str(row.get("risk_flags", "")) else "acceptable",
        "atr_pct": row.get("atr_pct"),
        "rsi_14": row.get("rsi_14"),
        "beta": row.get("beta"),
        "trend_status": row.get("trend_status"),
        "setup_type": row.get("setup_type"),
        "strategy_name": row.get("strategy"),
        "base_strategy_score": row.get("raw_strategy_score"),
        "advanced_technical_score": row.get("advanced_technical_score"),
        "higher_timeframe_context_score": row.get("context_score"),
        "deterministic_final_score": row.get("final_strategy_score"),
        "strategy_reasons": row.get("reasons"),
        "strategy_warnings": row.get("warnings"),
        "risk_flags": row.get("risk_flags"),
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "nearest_support": row.get("nearest_support"),
        "nearest_resistance": row.get("nearest_resistance"),
        "breakout_level": row.get("breakout_level"),
        "acceptance_status": row.get("breakout_acceptance_status"),
        "liquidity_sweep_status": row.get("liquidity_sweep_status"),
        "fair_value_gap_status": row.get("fvg_lvn_status"),
        "order_block_status": row.get("ob_hvn_status"),
        "volume_profile_status": row.get("volume_profile_location"),
        "vwap_status": row.get("vwap_status"),
        "entry_plan": {"entry_type": row.get("entry_type"), "entry_zone": row.get("entry_zone"), "trigger": row.get("entry_trigger")},
        "stop_plan": {"stop_price": row.get("stop_price"), "reason": row.get("stop_reason")},
        "target_plan": {"target_1": row.get("target_1"), "target_2": row.get("target_2"), "target_3": row.get("target_3")},
        "risk_reward": row.get("risk_reward_target_1"),
        "data_quality_score": row.get("data_quality_score"),
        "missing_data_fields": missing[:30],
        "market_context": {"market_regime": row.get("market_regime")},
        "competing_strategy_scores": competing_scores,
    }


def packet_hash(packet: dict[str, Any]) -> str:
    raw = json.dumps(packet, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(model: str, digest: str) -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    safe_model = model.replace("/", "_").replace(":", "_")
    return AI_LOG_DIR / f"{today}_{safe_model}_{AI_REVIEW_PROMPT_VERSION}_{digest}.json"


def _result_from_validated(row: pd.Series, validated: dict[str, Any], raw_response: Any, available: bool, valid: bool, error: str) -> AIReviewResult:
    components = {
        key: int(validated.get(key, 0))
        for key in [
            "setup_coherence_score",
            "entry_quality_score",
            "stop_quality_score",
            "target_quality_score",
            "catalyst_alignment_score",
            "risk_clarity_score",
            "confirmation_clarity_score",
        ]
    }
    return AIReviewResult(
        ticker=str(row.get("ticker", "")),
        strategy_name=str(row.get("strategy", "")),
        ai_available=available,
        ai_valid=valid,
        setup_quality=validated.get("setup_quality", "Weak"),
        watchlist_action=validated.get("watchlist_action", "Wait"),
        ai_review_score=int(validated.get("ai_review_score", 0)),
        component_scores=components,
        best_strategy_fit=validated.get("best_strategy_fit", ""),
        strategy_fit_comment=validated.get("strategy_fit_comment", ""),
        trade_maturity=validated.get("trade_maturity", ""),
        entry_quality=validated.get("entry_quality", ""),
        stop_quality=validated.get("stop_quality", ""),
        target_quality=validated.get("target_quality", ""),
        main_reason=validated.get("main_reason", ""),
        confirmation_needed=validated.get("confirmation_needed", []),
        entry_comment=validated.get("entry_comment", ""),
        stop_comment=validated.get("stop_comment", ""),
        target_comment=validated.get("target_comment", ""),
        catalyst_interpretation=validated.get("catalyst_interpretation", "unclear"),
        catalyst_comment=validated.get("catalyst_comment", ""),
        main_risks=validated.get("main_risks", []),
        downgrade_reasons=validated.get("downgrade_reasons", []),
        missing_data_notes=validated.get("missing_data_notes", []),
        confidence=validated.get("confidence", "Low"),
        ai_score_adjustment=int(validated.get("ai_score_adjustment", 0)),
        raw_response=raw_response,
        error_message=error,
    )


def fallback_review(row: pd.Series, message: str = "AI unavailable") -> AIReviewResult:
    return _result_from_validated(row, {"main_reason": message}, {}, False, False, message)


def review_candidate_with_ai(row: pd.Series, settings: AIReviewSettings, competing: pd.DataFrame | None = None) -> AIReviewResult:
    if not settings.enabled:
        return fallback_review(row, "AI review disabled")
    if not check_ollama_available(settings.base_url, min(settings.timeout, 5)):
        return fallback_review(row, "Ollama unavailable; using deterministic scan only")

    packet = build_candidate_packet(row, competing)
    digest = packet_hash(packet)
    cache_file = _cache_path(settings.model, digest)
    if cache_file.exists() and not settings.force_rerun:
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            validated, valid, status = validate_ai_review(cached.get("validated_response"))
            return _result_from_validated(row, validated, cached.get("raw_ai_response", {}), True, valid, status)
        except Exception:
            pass

    prompt = build_candidate_review_prompt(packet)
    parsed, error, raw = call_ollama_json(prompt, model=settings.model, base_url=settings.base_url, timeout=settings.timeout)
    validated, valid, status = validate_ai_review(parsed)
    result = _result_from_validated(row, validated, parsed or raw, True, valid and error is None, error or status)
    log_ai_review(row, settings, packet, digest, result, validated)
    return result


def integrate_ai_score(row: pd.Series, result: AIReviewResult, settings: AIReviewSettings) -> float:
    deterministic = float(row.get("final_strategy_score") or 0)
    score = deterministic
    if settings.enabled and result.ai_valid and settings.allow_score_influence:
        score = deterministic * (1 - settings.ai_weight) + result.ai_review_score * settings.ai_weight
        delta = score - deterministic
        if delta > 5 and not settings.allow_positive_adjustment:
            score = deterministic
        elif delta > 5:
            score = deterministic + 5
        if deterministic - score > 15:
            score = deterministic - 15

    if settings.allow_caps and result.ai_valid:
        if result.watchlist_action == "Avoid for now":
            score = min(score, 60)
        if result.watchlist_action == "Wait":
            score = min(score, 68)
        if result.watchlist_action == "Conditional only":
            score = min(score, 78)
        if result.setup_quality == "Invalid":
            score = min(score, 55)
        text = " ".join([result.entry_quality, result.stop_quality, result.target_quality, " ".join(result.downgrade_reasons)]).lower()
        if "extended" in text:
            score = min(score, 75)
        if "poor risk" in text or "too close" in text:
            score = min(score, 70)
        if "too wide" in text or "not tied to structure" in text:
            score = min(score, 72)

    if deterministic < 50:
        score = min(score, 60)
    if float(row.get("data_quality_score") or 100) < 60:
        score = min(score, 60)
    if "Low Liquidity" in str(row.get("risk_flags", "")):
        score = min(score, 70)
    if "Breakout Rejected" in str(row.get("risk_flags", "")) and row.get("strategy") == "Catalyst Gap / Multi-Month Breakout":
        score = min(score, 65)
    return round(max(0, min(100, score)), 1)


def log_ai_review(row: pd.Series, settings: AIReviewSettings, packet: dict[str, Any], digest: str, result: AIReviewResult, validated: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ticker": row.get("ticker"),
        "strategy_name": row.get("strategy"),
        "deterministic_score": row.get("final_strategy_score"),
        "ai_review_score": result.ai_review_score,
        "final_watch_score": None,
        "ai_action": result.watchlist_action,
        "setup_quality": result.setup_quality,
        "model_name": settings.model,
        "prompt_version": AI_REVIEW_PROMPT_VERSION,
        "candidate_packet_hash": digest,
        "raw_ai_response": result.raw_response,
        "validated_response": validated,
        "validation_status": "valid" if result.ai_valid else result.error_message,
    }
    log_path = AI_LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def review_dataframe_with_ai(df: pd.DataFrame, settings: AIReviewSettings) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    eligible = df[df["final_strategy_score"] >= settings.min_deterministic_score].head(settings.candidate_limit)
    eligible_keys = set(zip(eligible["ticker"], eligible["strategy"]))
    for _, row in df.iterrows():
        combined = row.to_dict()
        if (row.get("ticker"), row.get("strategy")) in eligible_keys:
            competing = df[df["ticker"] == row.get("ticker")]
            result = review_candidate_with_ai(row, settings, competing)
            final_watch_score = integrate_ai_score(row, result, settings)
            combined.update(flatten_ai_result(result, final_watch_score, settings.show_raw_json))
        else:
            combined.update(flatten_ai_result(fallback_review(row, "Below AI review threshold"), row.get("final_strategy_score"), settings.show_raw_json))
        rows.append(combined)
    return pd.DataFrame(rows)


def flatten_ai_result(result: AIReviewResult, final_watch_score: float, include_raw: bool = False) -> dict[str, Any]:
    data = {
        "ai_available": result.ai_available,
        "ai_valid": result.ai_valid,
        "ai_review_score": result.ai_review_score,
        "final_watch_score": final_watch_score,
        "ai_action": result.watchlist_action,
        "setup_quality": result.setup_quality,
        "trade_maturity": result.trade_maturity,
        "entry_quality": result.entry_quality,
        "stop_quality": result.stop_quality,
        "target_quality": result.target_quality,
        "catalyst_interpretation": result.catalyst_interpretation,
        "confirmation_needed": "; ".join(result.confirmation_needed),
        "main_reason": result.main_reason,
        "entry_comment": result.entry_comment,
        "stop_comment": result.stop_comment,
        "target_comment": result.target_comment,
        "main_risks": "; ".join(result.main_risks),
        "downgrade_reasons": "; ".join(result.downgrade_reasons),
        "missing_data_notes": "; ".join(result.missing_data_notes),
        "ai_confidence": result.confidence,
        "ai_error_message": result.error_message,
    }
    if include_raw:
        data["raw_ai_json"] = json.dumps(result.raw_response, default=str)
    return data
