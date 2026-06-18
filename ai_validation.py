from typing import Any

ALLOWED_SETUP_QUALITY = {"Strong", "Developing", "Conditional", "Weak", "Invalid"}
ALLOWED_ACTIONS = {"Strong watch", "Watch closely", "Conditional only", "Wait", "Avoid for now"}
ALLOWED_CATALYST = {"positive", "negative", "neutral", "unclear", "none"}
ALLOWED_CONFIDENCE = {"High", "Medium", "Low"}

COMPONENT_LIMITS = {
    "setup_coherence_score": 25,
    "entry_quality_score": 20,
    "stop_quality_score": 15,
    "target_quality_score": 15,
    "catalyst_alignment_score": 10,
    "risk_clarity_score": 10,
    "confirmation_clarity_score": 5,
}


def clamp_number(value: Any, minimum: int, maximum: int) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        number = minimum
    return max(minimum, min(maximum, number))


def validate_ai_review(raw: dict[str, Any] | None) -> tuple[dict[str, Any], bool, str]:
    if not isinstance(raw, dict):
        return {}, False, "AI response was not valid JSON object"

    output = dict(raw)
    for key, maximum in COMPONENT_LIMITS.items():
        output[key] = clamp_number(output.get(key, 0), 0, maximum)

    computed = sum(output[key] for key in COMPONENT_LIMITS)
    output["ai_review_score"] = clamp_number(output.get("ai_review_score", computed), 0, 100)
    if abs(output["ai_review_score"] - computed) > 5:
        output["ai_review_score"] = computed

    output["setup_quality"] = output.get("setup_quality") if output.get("setup_quality") in ALLOWED_SETUP_QUALITY else "Weak"
    output["watchlist_action"] = output.get("watchlist_action") if output.get("watchlist_action") in ALLOWED_ACTIONS else "Wait"
    output["catalyst_interpretation"] = output.get("catalyst_interpretation") if output.get("catalyst_interpretation") in ALLOWED_CATALYST else "unclear"
    output["confidence"] = output.get("confidence") if output.get("confidence") in ALLOWED_CONFIDENCE else "Low"
    output["ai_score_adjustment"] = clamp_number(output.get("ai_score_adjustment", 0), -15, 5)

    for key in ["confirmation_needed", "main_risks", "downgrade_reasons", "missing_data_notes"]:
        value = output.get(key, [])
        output[key] = value if isinstance(value, list) else [str(value)]

    for key in [
        "best_strategy_fit",
        "strategy_fit_comment",
        "trade_maturity",
        "entry_quality",
        "stop_quality",
        "target_quality",
        "main_reason",
        "entry_comment",
        "stop_comment",
        "target_comment",
        "catalyst_comment",
    ]:
        output[key] = "" if output.get(key) is None else str(output.get(key, ""))

    return output, True, "valid"
