from __future__ import annotations

import json

import pandas as pd

from ollama_client import call_ollama_json, check_ollama_available


def review_backtest_with_ai(
    metrics: dict,
    trades: pd.DataFrame,
    *,
    base_url: str = "http://localhost:11434",
    model: str = "llama3.2:latest",
    timeout: int = 30,
) -> dict:
    if not check_ollama_available(base_url, min(timeout, 5)):
        return {"available": False, "summary": "Ollama unavailable.", "risks": [], "improvements": []}
    sample = trades.tail(30).to_dict(orient="records") if not trades.empty else []
    prompt = (
        "You are reviewing a swing trading strategy backtest. Use only the supplied metrics and trade sample. "
        "Do not claim this predicts future returns. Return JSON with keys: summary, main_strengths, main_weaknesses, "
        "overfit_risks, practical_improvements, confidence.\n\n"
        f"Metrics:\n{json.dumps(metrics, default=str, indent=2)}\n\n"
        f"Recent trades:\n{json.dumps(sample, default=str, indent=2)}"
    )
    parsed, error, raw = call_ollama_json(prompt, model=model, base_url=base_url, timeout=timeout)
    if not parsed:
        return {"available": False, "summary": f"AI review failed: {error}", "raw": raw}
    parsed["available"] = True
    return parsed
