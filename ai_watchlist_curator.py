import json
from typing import Any

import pandas as pd

from ai_prompt_templates import build_weekly_curator_prompt
from ollama_client import call_ollama_json, check_ollama_available


def fallback_curation(df: pd.DataFrame) -> dict[str, Any]:
    def rows_for(action: str) -> list[dict[str, str]]:
        filtered = df[df.get("ai_action", "").fillna("") == action] if "ai_action" in df else pd.DataFrame()
        return [
            {
                "ticker": str(row.get("ticker")),
                "strategy": str(row.get("strategy")),
                "reason": str(row.get("main_reason") or row.get("reasons") or ""),
                "key_confirmation": str(row.get("confirmation_needed") or row.get("entry_trigger") or ""),
                "main_risk": str(row.get("risk_flags") or ""),
            }
            for _, row in filtered.head(8).iterrows()
        ]

    return {
        "weekly_market_summary": "AI curator unavailable; grouped by deterministic/individual review fields.",
        "strong_watch": rows_for("Strong watch"),
        "watch_closely": rows_for("Watch closely"),
        "conditional_only": rows_for("Conditional only"),
        "do_not_chase": rows_for("Wait"),
        "avoid_for_now": rows_for("Avoid for now"),
        "theme_notes": [],
        "risk_notes": [],
    }


def curate_weekly_watchlist(df: pd.DataFrame, base_url: str, model: str, timeout: int = 45, limit: int = 20) -> dict[str, Any]:
    if df.empty or not check_ollama_available(base_url, min(timeout, 5)):
        return fallback_curation(df)
    candidates = []
    columns = [
        "ticker",
        "category",
        "strategy",
        "final_watch_score",
        "final_strategy_score",
        "ai_review_score",
        "ai_action",
        "setup_quality",
        "risk_flags",
        "main_reason",
        "confirmation_needed",
    ]
    for _, row in df.sort_values("final_watch_score", ascending=False).head(limit).iterrows():
        candidates.append({column: row.get(column) for column in columns if column in row})
    parsed, error, raw = call_ollama_json(build_weekly_curator_prompt(candidates), model=model, base_url=base_url, timeout=timeout)
    if error or not isinstance(parsed, dict):
        return fallback_curation(df)
    for key in ["strong_watch", "watch_closely", "conditional_only", "do_not_chase", "avoid_for_now", "theme_notes", "risk_notes"]:
        parsed[key] = parsed.get(key, []) if isinstance(parsed.get(key, []), list) else []
    parsed["raw_curator_json"] = json.dumps(parsed if parsed else raw, default=str)
    return parsed
