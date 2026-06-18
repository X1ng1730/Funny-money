import json
from typing import Any

import requests


SYSTEM_INSTRUCTIONS = (
    "You are not a financial advisor. Do not invent data. Only use the supplied structured data. "
    "If data is missing, say it is missing. Produce a concise bullish swing-trading watchlist explanation. "
    "Do not say buy or sell. Use watchlist language such as Strong watch, Watch closely, Conditional only, Wait, or Avoid for now. "
    "When discussing order blocks, volume profile, VWAP, fair value gaps, or order-flow, call them estimated/proxy signals "
    "because the supplied data is yfinance OHLCV, not Level 2, tick, or true volume-at-price data."
)


def _fallback(message: str = "AI unavailable") -> dict[str, Any]:
    return {
        "ai_summary": message,
        "catalyst_interpretation": "unclear",
        "setup_quality_comment": "",
        "entry_comment": "",
        "stop_comment": "",
        "target_comment": "",
        "main_risks": [],
        "watchlist_action": "Wait",
        "setup_quality": "Weak",
        "best_strategy_fit": "",
        "confirmation_needed": [],
        "advanced_confluence_comment": "",
        "confidence": "Low",
        "missing_data_notes": [message],
        "ai_quality_adjustment": 0,
    }


def check_ollama_available(base_url: str = "http://localhost:11434", timeout: int = 3) -> bool:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return response.ok
    except Exception:
        return False


def list_ollama_models(base_url: str = "http://localhost:11434", timeout: int = 5) -> list[str]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        return [model.get("name", "") for model in response.json().get("models", []) if model.get("name")]
    except Exception:
        return []


def safe_parse_json_response(response: str | dict) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(response, dict):
        return response, None
    try:
        return json.loads(response), None
    except Exception as error:
        text = str(response)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1]), None
            except Exception:
                pass
        return None, str(error)


def call_ollama_json(
    prompt: str,
    schema: dict | None = None,
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    timeout: int = 30,
) -> tuple[dict[str, Any] | None, str | None, str]:
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json" if schema is None else schema,
            "options": {"temperature": 0.15},
        }
        response = requests.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=timeout)
        response.raise_for_status()
        raw = response.json().get("response", "{}")
        parsed, error = safe_parse_json_response(raw)
        return parsed, error, raw
    except Exception as error:
        return None, str(error), ""


def call_ollama_text(
    prompt: str,
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    timeout: int = 30,
) -> tuple[str | None, str | None]:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", ""), None
    except Exception as error:
        return None, str(error)


def build_prompt(payload: dict[str, Any], allow_adjustment: bool = False) -> str:
    adjustment = (
        "You may include ai_quality_adjustment from -5 to +5 only when clearly justified by the supplied catalyst/news text. "
        "Default to 0."
        if allow_adjustment
        else "Set ai_quality_adjustment to 0."
    )
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"{adjustment}\n\n"
        "Return JSON with keys: ai_summary, catalyst_interpretation, setup_quality, best_strategy_fit, "
        "setup_quality_comment, confirmation_needed, entry_zone_comment, entry_comment, stop_comment, "
        "target_comment, main_risks, advanced_confluence_comment, watchlist_action, confidence, "
        "missing_data_notes, ai_quality_adjustment. Do not invent levels. If VWAP is unavailable, say intraday VWAP is unavailable.\n\n"
        f"Structured data:\n{json.dumps(payload, default=str, indent=2)}"
    )


def summarize_setup(
    payload: dict[str, Any],
    base_url: str = "http://localhost:11434",
    model: str = "llama3.1",
    timeout: int = 20,
    allow_adjustment: bool = False,
) -> dict[str, Any]:
    prompt = build_prompt(payload, allow_adjustment)
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json().get("response", "{}")
        parsed = json.loads(raw)
        fallback = _fallback("")
        fallback.update(parsed)
        adjustment = fallback.get("ai_quality_adjustment", 0)
        try:
            adjustment = int(adjustment)
        except Exception:
            adjustment = 0
        fallback["ai_quality_adjustment"] = max(-5, min(5, adjustment)) if allow_adjustment else 0
        return fallback
    except Exception as error:
        return _fallback(f"AI unavailable: {error}")
