import json
from typing import Any

AI_REVIEW_PROMPT_VERSION = "v1"

AI_REVIEW_SYSTEM_PROMPT = """
You are an AI swing-trading setup reviewer inside a stock analysis app. You are not a financial advisor.
You do not place trades. You must not invent prices, levels, news, indicators, catalysts, earnings dates,
or missing values. You must only use the structured data supplied by the app. If data is missing, say it is
missing. Your job is to review whether the technical setup is clean, conditional, too extended, weak, or
invalid. Use watchlist language only. Do not say "buy now" or "sell now." Do not guarantee outcomes.
Focus on setup quality, entry quality, stop quality, target quality, catalyst alignment, risk clarity,
and confirmation needed. Treat volume profile, VWAP, fair value gaps, order blocks, and order-flow as
estimated/proxy signals when they come from yfinance OHLCV.
""".strip()


def build_candidate_review_prompt(candidate_packet: dict[str, Any]) -> str:
    return (
        f"{AI_REVIEW_SYSTEM_PROMPT}\n\n"
        "Return JSON only using this schema:\n"
        "{\n"
        '"setup_quality":"Strong|Developing|Conditional|Weak|Invalid",\n'
        '"watchlist_action":"Strong watch|Watch closely|Conditional only|Wait|Avoid for now",\n'
        '"ai_review_score":0,\n'
        '"best_strategy_fit":"string",\n'
        '"strategy_fit_comment":"string",\n'
        '"setup_coherence_score":0,\n'
        '"entry_quality_score":0,\n'
        '"stop_quality_score":0,\n'
        '"target_quality_score":0,\n'
        '"catalyst_alignment_score":0,\n'
        '"risk_clarity_score":0,\n'
        '"confirmation_clarity_score":0,\n'
        '"trade_maturity":"Ready|Conditional|Early|Late|Invalid",\n'
        '"entry_quality":"Good|Acceptable|Too extended|Too close to resistance|Needs pullback|Needs breakout confirmation|Not enough confirmation|Not valid",\n'
        '"stop_quality":"Logical|Too tight|Too wide|Not tied to structure|Too close to noise|Below correct invalidation level|Needs ATR buffer",\n'
        '"target_quality":"Realistic|Aggressive|Too close|Poor risk/reward|Depends on breakout continuation|No clear resistance target",\n'
        '"main_reason":"string",\n'
        '"confirmation_needed":["string"],\n'
        '"entry_comment":"string",\n'
        '"stop_comment":"string",\n'
        '"target_comment":"string",\n'
        '"catalyst_interpretation":"positive|negative|neutral|unclear|none",\n'
        '"catalyst_comment":"string",\n'
        '"main_risks":["string"],\n'
        '"downgrade_reasons":["string"],\n'
        '"missing_data_notes":["string"],\n'
        '"ai_score_adjustment":0,\n'
        '"confidence":"High|Medium|Low"\n'
        "}\n\n"
        "Component max scores: setup 25, entry 20, stop 15, target 15, catalyst 10, risk 10, confirmation 5. "
        "ai_review_score must equal their sum. Be stricter on risky or extended setups than on positive catalysts.\n\n"
        f"Candidate packet:\n{json.dumps(candidate_packet, default=str, indent=2)}"
    )


def build_weekly_curator_prompt(candidates: list[dict[str, Any]]) -> str:
    return (
        f"{AI_REVIEW_SYSTEM_PROMPT}\n\n"
        "Curate only the supplied scanned candidates. Do not invent tickers, news, or levels. Return JSON only:\n"
        "{"
        '"weekly_market_summary":"string",'
        '"strong_watch":[{"ticker":"string","strategy":"string","reason":"string","key_confirmation":"string","main_risk":"string"}],'
        '"watch_closely":[],'
        '"conditional_only":[],'
        '"do_not_chase":[],'
        '"avoid_for_now":[],'
        '"theme_notes":[{"theme":"string","comment":"string"}],'
        '"risk_notes":["string"]'
        "}\n\n"
        f"Candidates:\n{json.dumps(candidates, default=str, indent=2)}"
    )
