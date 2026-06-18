from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_metrics import calculate_backtest_metrics
from data_yfinance import build_market_snapshot, get_price_data
from strategy_engine import STRATEGIES, result_to_flat_row, run_strategies


BACKTEST_DIR = Path("data") / "backtests"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BacktestConfig:
    tickers: list[str]
    strategy: str = "All Strategies"
    start_date: str | None = None
    end_date: str | None = None
    history_period: str = "5y"
    min_score: float = 65
    min_rvol: float = 0.8
    max_atr_pct: float = 14
    require_above_200sma: bool = False
    entry_rule: str = "Next open"
    stop_rule: str = "Strategy stop"
    target_rule: str = "Target 1"
    max_holding_days: int = 20
    initial_equity: float = 100000.0
    risk_per_trade_pct: float = 1.0
    commission_per_trade: float = 0.0
    slippage_pct: float = 0.05
    allow_overlapping_trades: bool = False
    use_cache: bool = True


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _period_for_dates(start_date: str | None) -> str:
    if not start_date:
        return "5y"
    try:
        years = max(1, int((datetime.now() - pd.to_datetime(start_date).to_pydatetime()).days / 365) + 1)
    except Exception:
        return "5y"
    if years <= 2:
        return "2y"
    if years <= 5:
        return "5y"
    return "10y"


def _entry_price(row: pd.Series, entry_rule: str) -> float | None:
    if entry_rule == "Next open":
        return _num(row.get("open"))
    if entry_rule == "Next close":
        return _num(row.get("close"))
    return _num(row.get("open"))


def _target_price(signal: pd.Series, entry: float, stop: float, target_rule: str) -> float:
    if target_rule == "2R":
        return entry + (entry - stop) * 2
    if target_rule == "Target 2":
        target = _num(signal.get("target_2"))
        return target if target and target > entry else entry + (entry - stop) * 2
    target = _num(signal.get("target_1"))
    return target if target and target > entry else entry + (entry - stop) * 1.5


def _stop_price(signal: pd.Series, entry: float, stop_rule: str) -> float | None:
    if stop_rule == "3% fixed":
        return entry * 0.97
    if stop_rule == "5% fixed":
        return entry * 0.95
    stop = _num(signal.get("stop_price"))
    if stop and stop < entry:
        return stop
    atr = _num(signal.get("atr_14")) or entry * 0.04
    return entry - atr


def _passes_filters(signal: pd.Series, config: BacktestConfig) -> bool:
    if _num(signal.get("final_strategy_score")) is None or float(signal.get("final_strategy_score")) < config.min_score:
        return False
    if _num(signal.get("relative_volume")) is not None and float(signal.get("relative_volume")) < config.min_rvol:
        return False
    if _num(signal.get("atr_pct")) is not None and float(signal.get("atr_pct")) > config.max_atr_pct:
        return False
    if config.require_above_200sma and _num(signal.get("current_price")) is not None and _num(signal.get("sma_200")) is not None:
        if float(signal.get("current_price")) < float(signal.get("sma_200")):
            return False
    if "Low Liquidity" in str(signal.get("risk_flags", "")):
        return False
    return True


def _simulate_trade(price_data: pd.DataFrame, signal_index: int, signal: pd.Series, config: BacktestConfig, equity: float) -> dict | None:
    entry_index = signal_index + 1
    if entry_index >= len(price_data):
        return None
    entry_bar = price_data.iloc[entry_index]
    entry = _entry_price(entry_bar, config.entry_rule)
    if entry is None or entry <= 0:
        return None
    entry *= 1 + (config.slippage_pct / 100)
    stop = _stop_price(signal, entry, config.stop_rule)
    if stop is None or stop >= entry:
        return None
    target = _target_price(signal, entry, stop, config.target_rule)
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return None
    risk_dollars = equity * (config.risk_per_trade_pct / 100)
    shares = max(1, int(risk_dollars / risk_per_share))
    exit_price = None
    exit_reason = "Time exit"
    exit_index = min(len(price_data) - 1, entry_index + config.max_holding_days)
    for cursor in range(entry_index, exit_index + 1):
        bar = price_data.iloc[cursor]
        low = _num(bar.get("low"))
        high = _num(bar.get("high"))
        if low is not None and low <= stop:
            exit_price = stop * (1 - config.slippage_pct / 100)
            exit_reason = "Stop"
            exit_index = cursor
            break
        if high is not None and high >= target:
            exit_price = target * (1 - config.slippage_pct / 100)
            exit_reason = "Target"
            exit_index = cursor
            break
    if exit_price is None:
        exit_price = _num(price_data.iloc[exit_index].get("close"))
    if exit_price is None:
        return None
    pnl = (exit_price - entry) * shares - config.commission_per_trade
    r_multiple = (exit_price - entry) / risk_per_share
    return {
        "ticker": signal.get("ticker"),
        "strategy": signal.get("strategy"),
        "signal_date": price_data.index[signal_index].date().isoformat(),
        "entry_date": price_data.index[entry_index].date().isoformat(),
        "exit_date": price_data.index[exit_index].date().isoformat(),
        "entry_price": round(entry, 2),
        "stop_price": round(stop, 2),
        "target_price": round(target, 2),
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "holding_days": int(exit_index - entry_index + 1),
        "shares": int(shares),
        "pnl_dollars": round(float(pnl), 2),
        "return_pct": round(((exit_price - entry) / entry) * 100, 2),
        "r_multiple": round(float(r_multiple), 3),
        "signal_score": signal.get("final_strategy_score"),
        "match_label": signal.get("match_label"),
        "risk_flags": signal.get("risk_flags"),
        "reasons": signal.get("reasons"),
    }


def run_backtest(config: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    trades: list[dict] = []
    equity = float(config.initial_equity)
    last_exit_by_ticker: dict[str, pd.Timestamp] = {}
    for ticker in [item.upper().strip() for item in config.tickers if str(item).strip()]:
        try:
            data = get_price_data(ticker, period=config.history_period or _period_for_dates(config.start_date), interval="1d", use_cache=config.use_cache)
        except Exception:
            continue
        data = data.sort_index()
        start_ts = pd.to_datetime(config.start_date) if config.start_date else None
        end_ts = pd.to_datetime(config.end_date) if config.end_date else None
        if config.end_date:
            data = data[data.index <= end_ts]
        if len(data) < 240:
            continue
        for index in range(220, len(data) - 1):
            signal_date = data.index[index]
            if start_ts is not None and signal_date < start_ts:
                continue
            if not config.allow_overlapping_trades and ticker in last_exit_by_ticker and signal_date <= last_exit_by_ticker[ticker]:
                continue
            history = data.iloc[: index + 1].copy()
            try:
                snapshot = build_market_snapshot(ticker, history, {}, {})
                snapshot["active"] = True
                snapshot["category"] = "Backtest"
                snapshot["priority"] = "Backtest"
                row = pd.Series(snapshot)
                results = run_strategies(row, config.strategy)
                flattened = [result_to_flat_row(row, result) for result in results]
                if not flattened:
                    continue
                candidates = pd.DataFrame(flattened).sort_values("final_strategy_score", ascending=False)
                signal = candidates.iloc[0]
                if not _passes_filters(signal, config):
                    continue
                trade = _simulate_trade(data, index, signal, config, equity)
                if trade is None:
                    continue
                equity += trade["pnl_dollars"]
                trade["equity_after"] = round(equity, 2)
                trades.append(trade)
                last_exit_by_ticker[ticker] = pd.to_datetime(trade["exit_date"])
            except Exception:
                continue
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("exit_date").reset_index(drop=True)
    metrics = calculate_backtest_metrics(trades_df, config.initial_equity)
    return trades_df, metrics


def save_backtest_results(trades: pd.DataFrame, metrics: dict, config: BacktestConfig) -> Path:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = BACKTEST_DIR / f"{stamp}_backtest.csv"
    trades.to_csv(path, index=False)
    path.with_suffix(".json").write_text(
        pd.Series({"config": asdict(config), "metrics": metrics}).to_json(indent=2, default_handler=str),
        encoding="utf-8",
    )
    return path
