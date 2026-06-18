from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any
import json

import pandas as pd

from data_yfinance import get_price_data
from ollama_client import call_ollama_json, check_ollama_available


JOURNAL_DIR = Path("data") / "journal"
SCREENSHOT_DIR = JOURNAL_DIR / "trade_screenshots"
PLANNED_TRADES_FILE = JOURNAL_DIR / "planned_trades.csv"
TRADES_FILE = JOURNAL_DIR / "trades.csv"
AI_PLAN_REVIEWS_FILE = JOURNAL_DIR / "ai_plan_reviews.jsonl"
AI_TRADE_REVIEWS_FILE = JOURNAL_DIR / "ai_trade_reviews.jsonl"
JOURNAL_LESSONS_FILE = JOURNAL_DIR / "journal_lessons.jsonl"
TRADE_SCORECARDS_FILE = JOURNAL_DIR / "trade_scorecards.jsonl"


STRATEGY_OPTIONS = [
    "Catalyst Gap / Multi-Month Breakout",
    "EMA Pullback Trend Continuation",
    "Reversal / Reclaim Setup",
    "Intraday Trend Join Confirmation",
    "Other",
]

SETUP_TYPE_OPTIONS = [
    "Breakout",
    "Pullback",
    "Reclaim",
    "Gap Continuation",
    "Liquidity Sweep Reclaim",
    "200 SMA Reclaim",
    "EMA Bounce",
    "News/Catalyst Momentum",
    "Other",
]

PLAN_STATUS_OPTIONS = ["Planned", "Active", "Closed", "Cancelled", "Skipped"]

EXIT_REASON_OPTIONS = [
    "Planned Target Hit",
    "Stop Loss Hit",
    "Manual Exit",
    "Panic Exit",
    "Took Profit Early",
    "Trailing Stop",
    "Lost Key EMA",
    "Lost Support",
    "Resistance Rejection",
    "News Risk",
    "Market Weakness",
    "Other",
]

MISTAKE_TAG_OPTIONS = [
    "No Mistake",
    "Chased Entry",
    "Entered Too Early",
    "Entered Too Late",
    "Panic Sold",
    "Anxiety Exit",
    "Took Profit Too Early",
    "Held Too Long",
    "Ignored Stop",
    "Moved Stop",
    "Oversized",
    "FOMO",
    "Poor Risk/Reward",
    "Ignored Resistance",
    "Ignored Market Weakness",
    "Did Not Follow Plan",
    "Other",
]

TRADE_GRADE_OPTIONS = ["A+", "A", "B", "C", "D", "F", "Not graded"]


PLANNED_COLUMNS = [
    "plan_id",
    "plan_status",
    "trade_id",
    "setup_date",
    "day_of_week_setup",
    "ticker",
    "strategy_name",
    "setup_type",
    "planned_entry_price",
    "planned_stop_loss",
    "planned_exit_price",
    "planned_shares",
    "gross_exposure",
    "planned_risk_per_share",
    "planned_total_risk",
    "planned_reward_per_share",
    "planned_total_reward",
    "planned_risk_reward_ratio",
    "planned_return_pct_to_target",
    "planned_loss_pct_to_stop",
    "max_loss_dollars",
    "max_gain_dollars",
    "ai_plan_score",
    "ai_plan_confidence",
    "ai_plan_feedback",
    "journal_warning_at_plan",
    "one_rule_to_follow_if_taken",
    "personal_edge_note",
    "execution_risk_note",
    "similar_past_trade_warning",
    "anxiety_exit_warning",
    "plan_following_warning",
    "ai_plan_confidence_adjusted_by_journal",
    "actual_entry_date",
    "actual_entry_day_of_week",
    "actual_buy_price",
    "actual_shares",
    "why_entered",
    "entry_diff_from_plan_pct",
    "actual_gross_exposure",
    "actual_planned_risk_dollars_based_on_actual_entry",
    "active_created_at",
    "created_at",
    "updated_at",
]

TRADE_COLUMNS = [
    "trade_id",
    "plan_id",
    "setup_date",
    "ticker",
    "strategy_name",
    "setup_type",
    "planned_entry_price",
    "planned_stop_loss",
    "planned_exit_price",
    "planned_shares",
    "planned_risk_reward_ratio",
    "planned_total_risk",
    "planned_total_reward",
    "ai_plan_score",
    "ai_plan_confidence",
    "ai_plan_feedback",
    "journal_warning_at_plan",
    "one_rule_to_follow_if_taken",
    "actual_entry_date",
    "actual_entry_day_of_week",
    "actual_buy_price",
    "actual_shares",
    "why_entered",
    "entry_diff_from_plan_pct",
    "actual_exit_date",
    "actual_exit_day_of_week",
    "actual_sell_price",
    "actual_exit_reason",
    "followed_plan",
    "why_exited",
    "mistake_tag",
    "lesson_learned",
    "notes",
    "trade_grade_self",
    "screenshot_path",
    "close_created_at",
    "gross_exposure",
    "gross_exit_value",
    "net_pnl",
    "pnl_per_share",
    "return_pct",
    "holding_period_days",
    "win_loss",
    "actual_risk_per_share",
    "actual_risk_dollars",
    "actual_r_multiple",
    "planned_vs_actual_entry_diff_pct",
    "planned_vs_actual_exit_diff_pct",
    "estimated_plan_pnl",
    "would_plan_have_done_better",
    "plan_following_pnl_difference",
    "target_hit_after_exit",
    "stop_hit_after_exit",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "mfe_r",
    "mae_r",
    "followed_plan_score",
    "execution_score_formula",
    "ai_trade_score",
    "ai_trade_grade",
    "ai_trade_summary",
    "ai_trade_lesson",
]


def ensure_journal_files() -> None:
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if not PLANNED_TRADES_FILE.exists():
        pd.DataFrame(columns=PLANNED_COLUMNS).to_csv(PLANNED_TRADES_FILE, index=False)
    if not TRADES_FILE.exists():
        pd.DataFrame(columns=TRADE_COLUMNS).to_csv(TRADES_FILE, index=False)
    for path in [AI_PLAN_REVIEWS_FILE, AI_TRADE_REVIEWS_FILE, JOURNAL_LESSONS_FILE, TRADE_SCORECARDS_FILE]:
        path.touch(exist_ok=True)


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    ensure_journal_files()
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=columns)
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df[columns + [column for column in df.columns if column not in columns]]


def load_planned_trades() -> pd.DataFrame:
    return _read_csv(PLANNED_TRADES_FILE, PLANNED_COLUMNS)


def load_trades() -> pd.DataFrame:
    return _read_csv(TRADES_FILE, TRADE_COLUMNS)


def save_planned_trades(df: pd.DataFrame) -> None:
    ensure_journal_files()
    for column in PLANNED_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df.to_csv(PLANNED_TRADES_FILE, index=False)


def save_trades(df: pd.DataFrame) -> None:
    ensure_journal_files()
    for column in TRADE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df.to_csv(TRADES_FILE, index=False)


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def _day_name(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.day_name()


def generate_plan_id(existing: pd.DataFrame | None = None) -> str:
    existing = load_planned_trades() if existing is None else existing
    prefix = f"PLAN-{datetime.now().strftime('%Y%m%d')}-"
    values = existing.get("plan_id", pd.Series(dtype=str)).dropna().astype(str)
    today_numbers = []
    for value in values[values.str.startswith(prefix)]:
        try:
            today_numbers.append(int(value.split("-")[-1]))
        except Exception:
            pass
    return f"{prefix}{(max(today_numbers) + 1 if today_numbers else 1):04d}"


def generate_trade_id(existing: pd.DataFrame | None = None) -> str:
    existing = load_trades() if existing is None else existing
    prefix = f"TRADE-{datetime.now().strftime('%Y%m%d')}-"
    values = existing.get("trade_id", pd.Series(dtype=str)).dropna().astype(str)
    today_numbers = []
    for value in values[values.str.startswith(prefix)]:
        try:
            today_numbers.append(int(value.split("-")[-1]))
        except Exception:
            pass
    return f"{prefix}{(max(today_numbers) + 1 if today_numbers else 1):04d}"


def calculate_plan_fields(entry: float, stop: float, target: float, shares: int) -> dict[str, Any]:
    risk_per_share = entry - stop
    reward_per_share = target - entry
    return {
        "gross_exposure": round(entry * shares, 2),
        "planned_risk_per_share": round(risk_per_share, 4),
        "planned_total_risk": round(risk_per_share * shares, 2),
        "planned_reward_per_share": round(reward_per_share, 4),
        "planned_total_reward": round(reward_per_share * shares, 2),
        "planned_risk_reward_ratio": round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else None,
        "planned_return_pct_to_target": round((reward_per_share / entry) * 100, 2) if entry > 0 else None,
        "planned_loss_pct_to_stop": round((risk_per_share / entry) * 100, 2) if entry > 0 else None,
        "max_loss_dollars": round(risk_per_share * shares, 2),
        "max_gain_dollars": round(reward_per_share * shares, 2),
    }


def validate_plan(ticker: str, setup_date: Any, strategy_name: str, setup_type: str, entry: float, stop: float, target: float, shares: int) -> list[str]:
    errors = []
    if not str(ticker).strip():
        errors.append("Ticker is required.")
    if not _date_text(setup_date):
        errors.append("Setup date is required.")
    if not strategy_name:
        errors.append("Strategy is required.")
    if not setup_type:
        errors.append("Setup type is required.")
    if entry <= 0:
        errors.append("Planned entry price must be greater than 0.")
    if stop <= 0:
        errors.append("Planned stop loss must be greater than 0.")
    if target <= 0:
        errors.append("Planned exit / target price must be greater than 0.")
    if shares <= 0:
        errors.append("Planned shares must be greater than 0.")
    if stop >= entry:
        errors.append("Planned stop loss must be below planned entry for long stock trades.")
    if target <= entry:
        errors.append("Planned exit / target should be above planned entry for bullish stock trades.")
    return errors


def plan_warnings(plan: dict[str, Any], account_risk_limit: float | None = None) -> list[str]:
    warnings = []
    rr = _num(plan.get("planned_risk_reward_ratio"))
    total_risk = _num(plan.get("planned_total_risk"))
    loss_pct = _num(plan.get("planned_loss_pct_to_stop"))
    if rr is not None and rr < 1.5:
        warnings.append("Risk/reward is weak.")
    if rr is not None and rr >= 2:
        warnings.append("Risk/reward is acceptable.")
    if account_risk_limit and total_risk and total_risk > account_risk_limit:
        warnings.append("Planned risk is above your account risk setting.")
    if loss_pct is not None and loss_pct < 1:
        warnings.append("Stop is extremely close to entry.")
    if loss_pct is not None and loss_pct > 12:
        warnings.append("Stop is extremely far below entry.")
    return warnings


def create_planned_trade(
    setup_date: Any,
    ticker: str,
    strategy_name: str,
    setup_type: str,
    planned_entry_price: float,
    planned_stop_loss: float,
    planned_exit_price: float,
    planned_shares: int,
) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    setup_text = _date_text(setup_date)
    record = {
        "plan_id": generate_plan_id(),
        "plan_status": "Planned",
        "trade_id": "",
        "setup_date": setup_text,
        "day_of_week_setup": _day_name(setup_text),
        "ticker": ticker.upper().strip(),
        "strategy_name": strategy_name,
        "setup_type": setup_type,
        "planned_entry_price": float(planned_entry_price),
        "planned_stop_loss": float(planned_stop_loss),
        "planned_exit_price": float(planned_exit_price),
        "planned_shares": int(planned_shares),
        "ai_plan_score": None,
        "ai_plan_confidence": "",
        "ai_plan_feedback": "",
        "journal_warning_at_plan": "",
        "one_rule_to_follow_if_taken": "",
        "created_at": now,
        "updated_at": now,
    }
    record.update(calculate_plan_fields(float(planned_entry_price), float(planned_stop_loss), float(planned_exit_price), int(planned_shares)))
    return record


def append_planned_trade(record: dict[str, Any]) -> None:
    df = load_planned_trades()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    save_planned_trades(df)


def update_planned_trade(plan_id: str, updates: dict[str, Any]) -> None:
    df = load_planned_trades()
    if df.empty or "plan_id" not in df:
        return
    mask = df["plan_id"].astype(str) == str(plan_id)
    if not mask.any():
        return
    updates["updated_at"] = datetime.now().isoformat(timespec="seconds")
    for key, value in updates.items():
        if key not in df.columns:
            df[key] = pd.NA
        df.loc[mask, key] = value
    save_planned_trades(df)


def delete_planned_trade(plan_id: str) -> None:
    df = load_planned_trades()
    if df.empty:
        return
    save_planned_trades(df[df["plan_id"].astype(str) != str(plan_id)].copy())


def actualize_trade(plan_id: str, actual_entry_date: Any, actual_buy_price: float, actual_shares: int, why_entered: str = "") -> tuple[bool, str]:
    plans = load_planned_trades()
    row = plans[plans["plan_id"].astype(str) == str(plan_id)]
    if row.empty:
        return False, "Planned trade not found."
    plan = row.iloc[0]
    entry = _num(plan.get("planned_entry_price"))
    stop = _num(plan.get("planned_stop_loss"))
    buy = _num(actual_buy_price)
    shares = _int(actual_shares)
    if buy is None or buy <= 0:
        return False, "Actual buy price must be greater than 0."
    if shares is None or shares <= 0:
        return False, "Actual shares must be greater than 0."
    actual_date = _date_text(actual_entry_date)
    if not actual_date:
        return False, "Actual entry date is required."
    updates = {
        "plan_status": "Active",
        "actual_entry_date": actual_date,
        "actual_entry_day_of_week": _day_name(actual_date),
        "actual_buy_price": buy,
        "actual_shares": shares,
        "why_entered": why_entered,
        "entry_diff_from_plan_pct": round(((buy - entry) / entry) * 100, 2) if entry else None,
        "actual_gross_exposure": round(buy * shares, 2),
        "actual_planned_risk_dollars_based_on_actual_entry": round((buy - stop) * shares, 2) if stop and buy > stop else None,
        "active_created_at": datetime.now().isoformat(timespec="seconds"),
    }
    update_planned_trade(plan_id, updates)
    return True, "Trade moved to Active Trades."


def followed_plan_score(value: str) -> float:
    return {"Yes": 1.0, "Partially": 0.5, "No": 0.0, "No Plan": 0.0}.get(str(value), 0.0)


def calculate_closed_trade_fields(plan: pd.Series | dict[str, Any], close_data: dict[str, Any]) -> dict[str, Any]:
    entry = _num(plan.get("actual_buy_price"))
    sell = _num(close_data.get("actual_sell_price"))
    shares = _int(plan.get("actual_shares"))
    planned_entry = _num(plan.get("planned_entry_price"))
    planned_stop = _num(plan.get("planned_stop_loss"))
    planned_exit = _num(plan.get("planned_exit_price"))
    entry_date = _date_text(plan.get("actual_entry_date"))
    exit_date = _date_text(close_data.get("actual_exit_date"))
    net_pnl = (sell - entry) * shares if entry is not None and sell is not None and shares else None
    gross_exposure = entry * shares if entry is not None and shares else None
    actual_risk_per_share = entry - planned_stop if entry is not None and planned_stop is not None and planned_stop < entry else None
    actual_risk_dollars = actual_risk_per_share * shares if actual_risk_per_share is not None and shares else None
    holding_days = None
    try:
        holding_days = (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days
    except Exception:
        pass
    return {
        "actual_exit_day_of_week": _day_name(exit_date),
        "gross_exposure": round(gross_exposure, 2) if gross_exposure is not None else None,
        "gross_exit_value": round(sell * shares, 2) if sell is not None and shares else None,
        "net_pnl": round(net_pnl, 2) if net_pnl is not None else None,
        "pnl_per_share": round(sell - entry, 4) if sell is not None and entry is not None else None,
        "return_pct": round((net_pnl / gross_exposure) * 100, 2) if net_pnl is not None and gross_exposure else None,
        "holding_period_days": holding_days,
        "win_loss": "Win" if net_pnl and net_pnl > 0 else "Loss" if net_pnl and net_pnl < 0 else "Breakeven",
        "actual_risk_per_share": round(actual_risk_per_share, 4) if actual_risk_per_share is not None else None,
        "actual_risk_dollars": round(actual_risk_dollars, 2) if actual_risk_dollars is not None else None,
        "actual_r_multiple": round(net_pnl / actual_risk_dollars, 2) if net_pnl is not None and actual_risk_dollars and actual_risk_dollars > 0 else None,
        "planned_vs_actual_entry_diff_pct": round(((entry - planned_entry) / planned_entry) * 100, 2) if entry is not None and planned_entry else None,
        "planned_vs_actual_exit_diff_pct": round(((sell - planned_exit) / planned_exit) * 100, 2) if sell is not None and planned_exit else None,
        "followed_plan_score": followed_plan_score(close_data.get("followed_plan")),
    }


def execution_score_formula(trade: dict[str, Any]) -> int:
    score = 0
    rr = _num(trade.get("planned_risk_reward_ratio"))
    entry_diff = abs(_num(trade.get("planned_vs_actual_entry_diff_pct")) or 0)
    planned_shares = _num(trade.get("planned_shares"))
    actual_shares = _num(trade.get("actual_shares"))
    mistake = str(trade.get("mistake_tag") or "")
    if trade.get("plan_id") and _num(trade.get("planned_entry_price")) and _num(trade.get("planned_stop_loss")) and _num(trade.get("planned_exit_price")):
        score += 15
    if rr is not None and rr >= 2:
        score += 10
    elif rr is not None and rr >= 1.5:
        score += 6
    if entry_diff <= 1:
        score += 10
    elif entry_diff <= 3:
        score += 6
    if actual_shares is not None and planned_shares is not None and actual_shares <= planned_shares:
        score += 5
    score += int(followed_plan_score(trade.get("followed_plan")) * 20)
    if mistake not in {"Ignored Stop", "Moved Stop"}:
        score += 15
    if _num(trade.get("mfe_r")) is not None and _num(trade.get("actual_r_multiple")) is not None:
        mfe = _num(trade.get("mfe_r")) or 0
        actual_r = _num(trade.get("actual_r_multiple")) or 0
        score += 10 if actual_r >= min(2, mfe * 0.6) else 5 if actual_r > 0 else 2
    if str(trade.get("lesson_learned") or "").strip() and str(trade.get("why_exited") or "").strip():
        score += 5
    score += 10
    penalties = {
        "Took Profit Too Early": 8,
        "Panic Sold": 18,
        "Anxiety Exit": 18,
        "Chased Entry": 10,
        "Entered Too Early": 10,
        "Entered Too Late": 10,
        "Poor Risk/Reward": 10,
        "FOMO": 10,
        "Did Not Follow Plan": 15,
        "Moved Stop": 20,
        "Ignored Stop": 25,
        "Oversized": 20,
        "Held Too Long": 15,
    }
    score -= penalties.get(mistake, 0)
    return max(0, min(100, int(round(score))))


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_journal_files()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def personal_edge_context(strategy_name: str = "", setup_type: str = "") -> dict[str, Any]:
    trades = load_trades()
    if trades.empty:
        return {"note": "Not enough personal trade history for this strategy yet."}
    subset = trades.copy()
    if strategy_name:
        subset = subset[subset["strategy_name"].astype(str) == str(strategy_name)]
    if setup_type:
        subset = subset[subset["setup_type"].astype(str) == str(setup_type)]
    if len(subset) < 10:
        return {"note": "Not enough personal trade history for this strategy yet.", "sample_size": int(len(subset))}
    return {
        "sample_size": int(len(subset)),
        "avg_r": round(pd.to_numeric(subset["actual_r_multiple"], errors="coerce").mean(), 2),
        "avg_pnl": round(pd.to_numeric(subset["net_pnl"], errors="coerce").mean(), 2),
        "plan_follow_rate": round((subset["followed_plan"].astype(str) == "Yes").mean() * 100, 1),
        "avg_formula_score": round(pd.to_numeric(subset["execution_score_formula"], errors="coerce").mean(), 1),
        "most_common_mistake": subset["mistake_tag"].mode().iloc[0] if not subset["mistake_tag"].dropna().empty else "",
        "early_exit_frequency": round(subset["mistake_tag"].astype(str).isin(["Panic Sold", "Anxiety Exit", "Took Profit Too Early"]).mean() * 100, 1),
    }


def run_ai_plan_review(plan: dict[str, Any] | pd.Series, base_url: str, model: str, timeout: int = 30, use_personal_history: bool = True) -> dict[str, Any]:
    if not check_ollama_available(base_url, min(timeout, 5)):
        return {
            "ai_plan_confidence": "Low",
            "ai_plan_score": None,
            "ai_plan_feedback": "Ollama unavailable; plan saved without AI review.",
            "journal_warning_at_plan": "",
            "one_rule_to_follow_if_taken": "",
        }
    personal = personal_edge_context(plan.get("strategy_name", ""), plan.get("setup_type", "")) if use_personal_history else {}
    prompt = (
        "You are reviewing a planned long-only stock swing trade before entry. Do not say buy or sell. "
        "Use educational watchlist language only. Return valid JSON with keys: ai_plan_confidence, ai_plan_score, "
        "setup_quality, risk_reward_assessment, entry_quality, stop_quality, target_quality, main_strengths, main_risks, "
        "confirmation_needed, journal_warning, one_rule_to_follow_if_taken, avoid_trade_reason, confidence_explanation, "
        "missing_data_notes, personal_edge_note, execution_risk_note, similar_past_trade_warning, anxiety_exit_warning, "
        "plan_following_warning, ai_plan_confidence_adjusted_by_journal.\n\n"
        f"Planned trade:\n{json.dumps(dict(plan), default=str, indent=2)}\n\n"
        f"Personal history context:\n{json.dumps(personal, default=str, indent=2)}"
    )
    parsed, error, raw = call_ollama_json(prompt, model=model, base_url=base_url, timeout=timeout)
    if not parsed:
        parsed = {
            "ai_plan_confidence": "Low",
            "ai_plan_score": None,
            "journal_warning": f"AI review failed: {error}",
            "one_rule_to_follow_if_taken": "",
            "missing_data_notes": [raw or str(error)],
        }
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "plan_id": plan.get("plan_id"),
        "ticker": plan.get("ticker"),
        "model": model,
        "review": parsed,
    }
    _append_jsonl(AI_PLAN_REVIEWS_FILE, record)
    return {
        "ai_plan_confidence": parsed.get("ai_plan_confidence") or parsed.get("confidence") or "Low",
        "ai_plan_score": parsed.get("ai_plan_score"),
        "ai_plan_feedback": parsed.get("confidence_explanation") or parsed.get("avoid_trade_reason") or parsed.get("journal_warning") or "",
        "journal_warning_at_plan": parsed.get("journal_warning", ""),
        "one_rule_to_follow_if_taken": parsed.get("one_rule_to_follow_if_taken", ""),
        "personal_edge_note": parsed.get("personal_edge_note", personal.get("note", "")),
        "execution_risk_note": parsed.get("execution_risk_note", ""),
        "similar_past_trade_warning": parsed.get("similar_past_trade_warning", ""),
        "anxiety_exit_warning": parsed.get("anxiety_exit_warning", ""),
        "plan_following_warning": parsed.get("plan_following_warning", ""),
        "ai_plan_confidence_adjusted_by_journal": parsed.get("ai_plan_confidence_adjusted_by_journal", ""),
    }


def save_uploaded_screenshot(uploaded_file: Any, trade_id: str) -> str:
    if uploaded_file is None:
        return ""
    ensure_journal_files()
    suffix = Path(uploaded_file.name).suffix or ".png"
    path = SCREENSHOT_DIR / f"{trade_id}{suffix}"
    path.write_bytes(uploaded_file.getbuffer())
    return str(path)


def build_trade_hindsight_packet(trade: dict[str, Any] | pd.Series, lookforward_days: int = 10, use_cache: bool = True) -> dict[str, Any]:
    packet = dict(trade)
    ticker = str(packet.get("ticker") or "").upper().strip()
    if not ticker:
        packet["hindsight_status"] = "Missing ticker."
        return packet
    try:
        data = get_price_data(ticker, period="5y", interval="1d", use_cache=use_cache).sort_index()
    except Exception as error:
        packet["hindsight_status"] = f"Historical data unavailable: {error}"
        return packet
    entry_date = pd.to_datetime(packet.get("actual_entry_date"), errors="coerce")
    exit_date = pd.to_datetime(packet.get("actual_exit_date"), errors="coerce")
    if pd.isna(entry_date) or pd.isna(exit_date):
        packet["hindsight_status"] = "Entry or exit date missing."
        return packet
    holding = data[(data.index >= entry_date) & (data.index <= exit_date)]
    post_exit = data[data.index > exit_date].head(lookforward_days)
    entry = _num(packet.get("actual_buy_price"))
    stop = _num(packet.get("planned_stop_loss"))
    target = _num(packet.get("planned_exit_price"))
    shares = _int(packet.get("actual_shares")) or 0
    risk_per_share = _num(packet.get("actual_risk_per_share"))
    if holding.empty or entry is None:
        packet["hindsight_status"] = "No daily candles available for holding period."
        return packet
    highest = _num(holding["high"].max())
    lowest = _num(holding["low"].min())
    mfe = highest - entry if highest is not None else None
    mae = entry - lowest if lowest is not None else None
    target_hit = bool(target is not None and highest is not None and highest >= target)
    stop_hit = bool(stop is not None and lowest is not None and lowest <= stop)
    target_after = bool(target is not None and not post_exit.empty and post_exit["high"].max() >= target)
    stop_after = bool(stop is not None and not post_exit.empty and post_exit["low"].min() <= stop)
    estimated_plan_pnl = None
    plan_outcome = "Unknown"
    uncertainty = ""
    if stop is not None and target is not None:
        future = data[data.index >= entry_date]
        for _, bar in future.iterrows():
            hit_target = bar.get("high") >= target
            hit_stop = bar.get("low") <= stop
            if hit_target and hit_stop:
                plan_outcome = "Mixed"
                uncertainty = "Daily candle hit target and stop; intraday sequence is uncertain."
                break
            if hit_target:
                plan_outcome = "Target first"
                estimated_plan_pnl = (target - entry) * shares
                break
            if hit_stop:
                plan_outcome = "Stop first"
                estimated_plan_pnl = (stop - entry) * shares
                break
    actual_pnl = _num(packet.get("net_pnl"))
    difference = estimated_plan_pnl - actual_pnl if estimated_plan_pnl is not None and actual_pnl is not None else None
    would_better = "Unknown"
    if difference is not None:
        would_better = "Yes" if difference > 0 else "No" if difference < 0 else "Mixed"
    packet.update(
        {
            "target_hit_during_trade": target_hit,
            "stop_hit_during_trade": stop_hit,
            "target_hit_after_exit": target_after,
            "stop_hit_after_exit": stop_after,
            "max_favorable_excursion": round(mfe, 4) if mfe is not None else None,
            "max_adverse_excursion": round(mae, 4) if mae is not None else None,
            "mfe_r": round(mfe / risk_per_share, 2) if mfe is not None and risk_per_share and risk_per_share > 0 else None,
            "mae_r": round(mae / risk_per_share, 2) if mae is not None and risk_per_share and risk_per_share > 0 else None,
            "estimated_plan_pnl": round(estimated_plan_pnl, 2) if estimated_plan_pnl is not None else None,
            "hindsight_plan_outcome": plan_outcome,
            "hindsight_uncertainty": uncertainty,
            "would_plan_have_done_better": would_better,
            "plan_following_pnl_difference": round(difference, 2) if difference is not None else None,
            "post_exit_lookforward_days": lookforward_days,
            "hindsight_status": "OK" if not uncertainty else uncertainty,
        }
    )
    return packet


def close_active_trade(
    plan_id: str,
    close_data: dict[str, Any],
    screenshot_path: str = "",
    run_ai: bool = False,
    base_url: str = "http://localhost:11434",
    model: str = "llama3.2:latest",
    timeout: int = 30,
    use_cache: bool = True,
) -> tuple[bool, str, str | None]:
    plans = load_planned_trades()
    plan_rows = plans[plans["plan_id"].astype(str) == str(plan_id)]
    if plan_rows.empty:
        return False, "Active planned trade not found.", None
    plan = plan_rows.iloc[0]
    if str(plan.get("plan_status")) != "Active":
        return False, "Only Active trades can be closed.", None
    exit_date = _date_text(close_data.get("actual_exit_date"))
    entry_date = _date_text(plan.get("actual_entry_date"))
    if not exit_date:
        return False, "Actual exit date is required.", None
    if entry_date and pd.to_datetime(exit_date) < pd.to_datetime(entry_date):
        return False, "Close date cannot be before entry date.", None
    sell = _num(close_data.get("actual_sell_price"))
    if sell is None or sell <= 0:
        return False, "Actual sell price must be greater than 0.", None
    trades = load_trades()
    trade_id = generate_trade_id(trades)
    if screenshot_path:
        try:
            old_path = Path(screenshot_path)
            if old_path.exists() and not old_path.stem.startswith(trade_id):
                new_path = old_path.with_name(f"{trade_id}{old_path.suffix}")
                old_path.replace(new_path)
                screenshot_path = str(new_path)
        except Exception:
            pass
    record = {column: plan.get(column, "") for column in TRADE_COLUMNS}
    record.update(
        {
            "trade_id": trade_id,
            "plan_id": plan_id,
            "actual_exit_date": exit_date,
            "actual_sell_price": sell,
            "actual_exit_reason": close_data.get("actual_exit_reason"),
            "followed_plan": close_data.get("followed_plan"),
            "why_exited": close_data.get("why_exited", ""),
            "mistake_tag": close_data.get("mistake_tag", ""),
            "lesson_learned": close_data.get("lesson_learned", ""),
            "notes": close_data.get("notes", ""),
            "trade_grade_self": close_data.get("trade_grade_self", ""),
            "screenshot_path": screenshot_path,
            "close_created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    record.update(calculate_closed_trade_fields(plan, close_data))
    record = build_trade_hindsight_packet(record, use_cache=use_cache)
    record["execution_score_formula"] = execution_score_formula(record)
    if run_ai:
        review = run_ai_trade_review(record, base_url=base_url, model=model, timeout=timeout)
        record.update(
            {
                "ai_trade_score": review.get("ai_trade_score"),
                "ai_trade_grade": review.get("ai_trade_grade"),
                "ai_trade_summary": review.get("summary", ""),
                "ai_trade_lesson": review.get("lesson", ""),
            }
        )
    trades = pd.concat([trades, pd.DataFrame([record])], ignore_index=True)
    save_trades(trades)
    update_planned_trade(plan_id, {"plan_status": "Closed", "trade_id": trade_id})
    _append_jsonl(TRADE_SCORECARDS_FILE, {"timestamp": datetime.now().isoformat(timespec="seconds"), "trade_id": trade_id, "formula_score": record.get("execution_score_formula")})
    if record.get("lesson_learned"):
        _append_jsonl(JOURNAL_LESSONS_FILE, {"timestamp": datetime.now().isoformat(timespec="seconds"), "trade_id": trade_id, "lesson": record.get("lesson_learned"), "mistake_tag": record.get("mistake_tag")})
    return True, "Trade closed and added to Closed Trade Journal.", trade_id


def run_ai_trade_review(trade: dict[str, Any] | pd.Series, base_url: str, model: str, timeout: int = 30) -> dict[str, Any]:
    if not check_ollama_available(base_url, min(timeout, 5)):
        return {"ai_trade_grade": "", "ai_trade_score": None, "summary": "Ollama unavailable.", "lesson": ""}
    prompt = (
        "You are grading a completed long-only stock swing trade in hindsight. Grade process, discipline, risk management, "
        "plan adherence, and execution; do not grade only by P&L. Use educational language. Return JSON with keys: "
        "ai_trade_grade, ai_trade_score, process_score, plan_adherence_score, entry_execution_score, exit_execution_score, "
        "risk_management_score, setup_quality_score, reflection_quality_score, pnl_context_score, planned_vs_actual_assessment, "
        "would_plan_have_done_better, judgment_call_assessment, summary, grade_reason, what_went_well, what_went_wrong, "
        "entry_assessment, exit_assessment, risk_management_assessment, plan_following_assessment, hindsight_chart_assessment, "
        "main_mistake, lesson, one_rule_for_next_time, strategy_adjustment_suggestion, tags_to_add, confidence, missing_data_notes.\n\n"
        f"Trade packet:\n{json.dumps(dict(trade), default=str, indent=2)}"
    )
    parsed, error, raw = call_ollama_json(prompt, model=model, base_url=base_url, timeout=timeout)
    if not parsed:
        parsed = {"ai_trade_grade": "", "ai_trade_score": None, "summary": f"AI review failed: {error}", "lesson": "", "raw": raw}
    _append_jsonl(AI_TRADE_REVIEWS_FILE, {"timestamp": datetime.now().isoformat(timespec="seconds"), "trade_id": trade.get("trade_id"), "model": model, "review": parsed})
    return parsed


def pnl_summary(trades: pd.DataFrame | None = None) -> dict[str, Any]:
    trades = load_trades() if trades is None else trades
    if trades.empty:
        return {}
    pnl = pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    mistake = trades["mistake_tag"].fillna("")
    profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else None
    return {
        "Total Net P&L": round(float(pnl.sum()), 2),
        "Total Trades": int(len(trades)),
        "Winning Trades": int((pnl > 0).sum()),
        "Losing Trades": int((pnl < 0).sum()),
        "Breakeven Trades": int((pnl == 0).sum()),
        "Win Rate": round(float((pnl > 0).mean() * 100), 1),
        "Average Win": round(float(wins.mean()), 2) if not wins.empty else 0,
        "Average Loss": round(float(losses.mean()), 2) if not losses.empty else 0,
        "Largest Win": round(float(pnl.max()), 2),
        "Largest Loss": round(float(pnl.min()), 2),
        "Profit Factor": round(float(profit_factor), 2) if profit_factor is not None else "N/A",
        "Average Return %": round(pd.to_numeric(trades["return_pct"], errors="coerce").mean(), 2),
        "Average R": round(pd.to_numeric(trades["actual_r_multiple"], errors="coerce").mean(), 2),
        "Plan Follow Rate": round((trades["followed_plan"].astype(str) == "Yes").mean() * 100, 1),
        "Average AI Grade": trades["ai_trade_grade"].dropna().astype(str).replace("", pd.NA).dropna().mode().iloc[0] if not trades["ai_trade_grade"].dropna().empty else "N/A",
        "Average Formula Score": round(pd.to_numeric(trades["execution_score_formula"], errors="coerce").mean(), 1),
        "Most Common Mistake": mistake.mode().iloc[0] if not mistake.dropna().empty else "N/A",
        "Most Costly Mistake": trades.assign(_pnl=pnl).groupby("mistake_tag")["_pnl"].sum().sort_values().index[0] if "mistake_tag" in trades and not trades.empty else "N/A",
        "Best Strategy by P&L": trades.assign(_pnl=pnl).groupby("strategy_name")["_pnl"].sum().sort_values(ascending=False).index[0] if "strategy_name" in trades and not trades.empty else "N/A",
        "Worst Strategy by P&L": trades.assign(_pnl=pnl).groupby("strategy_name")["_pnl"].sum().sort_values().index[0] if "strategy_name" in trades and not trades.empty else "N/A",
    }


def grouped_performance(trades: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if trades.empty or group_col not in trades:
        return pd.DataFrame()
    data = trades.copy()
    data["net_pnl"] = pd.to_numeric(data["net_pnl"], errors="coerce").fillna(0)
    data["actual_r_multiple"] = pd.to_numeric(data["actual_r_multiple"], errors="coerce")
    data["execution_score_formula"] = pd.to_numeric(data["execution_score_formula"], errors="coerce")
    grouped = data.groupby(group_col).agg(
        trades=("trade_id", "count"),
        net_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
        win_rate=("net_pnl", lambda value: (value > 0).mean() * 100),
        avg_r=("actual_r_multiple", "mean"),
        avg_formula_score=("execution_score_formula", "mean"),
    )
    return grouped.round(2).reset_index()


def planned_vs_actual_table(trades: pd.DataFrame | None = None) -> pd.DataFrame:
    trades = load_trades() if trades is None else trades
    if trades.empty:
        return pd.DataFrame()
    columns = {
        "ticker": "Ticker",
        "strategy_name": "Strategy",
        "actual_entry_date": "Entry Date",
        "actual_exit_date": "Exit Date",
        "planned_entry_price": "Planned Entry",
        "actual_buy_price": "Actual Buy",
        "planned_vs_actual_entry_diff_pct": "Entry Diff %",
        "planned_stop_loss": "Planned Stop",
        "planned_exit_price": "Planned Exit",
        "actual_sell_price": "Actual Sell",
        "planned_vs_actual_exit_diff_pct": "Exit Diff %",
        "net_pnl": "Actual P&L",
        "estimated_plan_pnl": "Estimated Plan P&L",
        "plan_following_pnl_difference": "Difference",
        "would_plan_have_done_better": "Would Plan Have Done Better?",
        "target_hit_after_exit": "Target Hit After Exit?",
        "stop_hit_after_exit": "Stop Hit After Exit?",
        "mistake_tag": "Mistake Tag",
        "ai_trade_grade": "AI Grade",
        "ai_trade_lesson": "AI Lesson",
    }
    visible = trades[[column for column in columns if column in trades.columns]].rename(columns=columns)
    for column in visible.select_dtypes(include="number").columns:
        visible[column] = visible[column].round(2)
    return visible


def journal_coach_summary(base_url: str, model: str, timeout: int = 30) -> dict[str, Any]:
    trades = load_trades()
    plans = load_planned_trades()
    if trades.empty:
        return {"period_summary": "No closed trades yet.", "confidence": "Low", "sample_size_warning": "Journal is empty."}
    if not check_ollama_available(base_url, min(timeout, 5)):
        return {"period_summary": "Ollama unavailable. Review the Personal Edge tables for now.", "confidence": "Low"}
    payload = {
        "summary": pnl_summary(trades),
        "strategy_edge": grouped_performance(trades, "strategy_name").to_dict(orient="records"),
        "mistake_edge": grouped_performance(trades, "mistake_tag").to_dict(orient="records"),
        "recent_trades": trades.tail(30).to_dict(orient="records"),
        "planned_trade_counts": plans["plan_status"].value_counts().to_dict() if not plans.empty else {},
    }
    prompt = (
        "You are a personal trading journal coach for long-only stock swing trading. Use only the supplied data. "
        "Return JSON with keys: period_summary, best_process_patterns, worst_process_patterns, best_pnl_patterns, "
        "worst_pnl_patterns, repeated_mistakes, most_costly_mistakes, planned_vs_actual_patterns, panic_exit_patterns, "
        "good_judgment_patterns, strategy_strengths, strategy_weaknesses, lucky_win_patterns, good_loss_patterns, "
        "rules_to_follow_next_week, setups_to_be_careful_with, scanner_feedback_suggestions, backtest_experiments_to_run, "
        "confidence, sample_size_warning.\n\n"
        f"Journal data:\n{json.dumps(payload, default=str, indent=2)}"
    )
    parsed, error, raw = call_ollama_json(prompt, model=model, base_url=base_url, timeout=timeout)
    return parsed or {"period_summary": f"AI journal coach failed: {error}", "raw": raw, "confidence": "Low"}
