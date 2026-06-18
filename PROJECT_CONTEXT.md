# Swing Trading Model - Project Context

This file is a handoff note for future VS Code / Codex / Copilot chats. It summarizes the current state of the project and the intended development direction.

## Project Location

`C:\Users\xingy\OneDrive\Desktop\Vscode Projects\Swing Trading Model`

## User Experience Level

The user is still learning Python, VS Code, virtual environments, Streamlit, Git, Codex/Copilot, and project structure.

When giving instructions:

- Explain steps clearly.
- Specify whether commands should be run in PowerShell or written inside a `.py` file.
- Do not assume the user knows whether they are in PowerShell versus the Python `>>>` shell.
- Work incrementally and preserve working milestones.
- Prefer making code changes directly when the request is implementation-oriented.

## Safety Rules

- Do not delete files unless the user explicitly asks.
- Do not hardcode API keys.
- Do not read or expose `.env` contents unless absolutely necessary and explicitly approved.
- Ask before installing packages.
- Ask before resetting Git.
- Do not present dashboard output as guaranteed predictions or direct buy/sell instructions.
- Keep all analysis risk-aware, cautious, and educational.
- This app is for personal long-only stock swing trading. No options, shorts, brokerage execution, or automated trading.
- Local folders under `data/`, `data_cache/`, and `data/journal/` may contain user working data. Do not overwrite or clear them unless explicitly requested.

## Current Tech Stack

- Windows
- VS Code
- Python virtual environment: `.venv`
- Streamlit for dashboard UI
- pandas for data manipulation
- yfinance as the main market data source
- Plotly for charts
- Ollama local AI, currently using `llama3.2:latest` in the UI by default
- Alpha Vantage helper module exists but is not central to the current workflow
- Git for local checkpoints

## Important Commands

Run these in PowerShell from the project folder.

Activate venv:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Open:

```text
http://127.0.0.1:8502
```

Stop Streamlit:

```text
Ctrl + C
```

Compile-check key modules:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py journal.py scanner.py scan_logger.py backtester.py backtest_metrics.py backtest_visuals.py backtest_ai_review.py strategy_presets.py
```

Check Git:

```powershell
git status --short
git log --oneline
```

## Current Project Goal

Build a personal swing-trading research and feedback-loop dashboard for stocks.

Target holding period is roughly 1-6 weeks. The app should help the user:

- Maintain a watchlist.
- Scan for swing setups.
- Review strategy candidates.
- View clean charts with relevant levels.
- Backtest strategy ideas.
- Plan trades before entry.
- Track active trades.
- Close and review trades.
- Learn from planned-vs-actual behavior and repeated mistakes.

The long-term goal is a feedback loop:

```text
Watchlist scan
  -> trade plan
  -> AI pre-trade review
  -> actual entry
  -> close trade
  -> planned-vs-actual review
  -> AI hindsight grade
  -> journal lessons
  -> better future trade planning
```

## Current Streamlit Pages

`app.py` is the main dashboard entry point.

Current pages:

- Single Stock Analysis
- Watchlist Ranking
- Watchlist Dashboard
- Strategy Scanner
- Comprehensive Scanner
- Weekly Trade Watchlist
- Backtesting Lab
- Trade Journal
- Chart Viewer

The app should be run through `app.py`; individual helper modules should not be run as separate apps.

## Current Architecture Overview

High-level flow:

```text
app.py
  -> watchlist_manager.py loads/saves watchlist metadata
  -> data_yfinance.py fetches/caches OHLCV and builds market snapshots
  -> indicators.py calculates indicators
  -> strategy_engine.py evaluates setup strategies
  -> trade_plan.py builds entry/stop/target plan levels
  -> scanner.py runs full watchlist scans
  -> backtester.py tests the same strategy engine historically
  -> journal.py manages planned/active/closed trade lifecycle
  -> ollama_client.py and ai_*.py modules provide optional local AI review
  -> charting.py renders Plotly charts
```

## Key Files And Modules

### `app.py`

Main Streamlit UI.

Important current additions:

- Sidebar Ollama settings.
- Persistent yfinance cache controls.
- Comprehensive Scanner page.
- Backtesting Lab page.
- Trade Journal page with 8 tabs.
- Cleaner Chart Viewer overlay controls.

### `data_yfinance.py`

Main yfinance data layer.

Current behavior:

- Uses `data_cache/` for ticker-period CSV cache.
- Uses `data/watchlist_market_cache.csv` for persistent watchlist market snapshot cache.
- Provides `get_price_data()`.
- Provides `get_multiple_price_data()`.
- Provides `get_watchlist_market_data()`.
- Provides `clear_watchlist_market_cache()`.
- Provides `build_market_snapshot()`, which is reused by the backtester to evaluate historical daily snapshots.
- Normalizes yfinance OHLCV columns to lowercase.

Important note: shell network access may be restricted in Codex, so yfinance/Yahoo requests can fail there even when the user browser/app environment works.

### `indicators.py`

Local indicator module. It now supports the broader app, including EMA/SMA, returns, volume averages, ATR, RSI, and helper functions such as `latest_complete_row()`.

### `ranking_model.py`

Starter ranking model for the Watchlist Ranking page.

This is separate from the richer strategy engine. It is still useful for quick simple ranking, but newer scanner/backtesting workflows primarily use `strategy_engine.py`.

### `strategy_engine.py`

Main strategy scoring system.

Current strategy families:

- Catalyst Gap / Multi-Month Breakout
- EMA Pullback Trend Continuation
- Reversal / Reclaim Setup

Important functions:

- `run_strategies(row, strategy_name=None)`
- `evaluate_strategy(row, strategy_name)`
- `result_to_flat_row(row, result)`
- `build_strategy_results(dashboard_df)`
- `best_strategy_per_ticker(strategy_df)`

The strategy engine combines:

- Raw strategy setup score.
- Advanced technical score.
- Higher-timeframe context score.
- Risk flags.
- Generated entry/stop/target plan.

### `trade_plan.py`

Generates planned trade levels for strategy results:

- Entry zone
- Entry trigger
- Stop price
- Invalidation
- Targets
- Risk/reward values

### Advanced Technical Modules

These modules provide estimated/proxy confluence from daily yfinance OHLCV data:

- `support_resistance.py`
- `volume_profile.py`
- `acceptance_rejection.py`
- `liquidity.py`
- `fair_value_gap.py`
- `order_blocks.py`
- `vwap.py`
- `confluence.py`

Important language: because the data is daily OHLCV from yfinance, volume profile, VWAP, fair value gaps, order blocks, and liquidity sweeps should be described as estimated/proxy signals.

### `charting.py`

Creates Plotly chart figures.

Current chart direction:

- Dark-mode friendly.
- Intentionally less cluttered.
- Default overlays:
  - 8 EMA in light purple.
  - 200 SMA in light pink.
  - White support/resistance lines.
  - Swing high/swing low support and resistance.
  - Trade plan take-profit zone and sell/invalidation zone.
- Advanced overlays are hidden under an optional expander.

### Scanner Modules

`scanner.py`

- Runs full watchlist strategy scans.
- Applies deterministic filters.
- Optionally calls AI review.
- Adds deterministic watch/action labels if AI is disabled.
- Summarizes scan results.

`scan_logger.py`

- Saves scan results to `data/scans/`.
- Loads latest/saved scan CSV files.

### Backtesting Modules

`backtester.py`

- Defines `BacktestConfig`.
- Runs daily-bar backtests using the same `strategy_engine.py`.
- Uses next-day entry after signal to reduce lookahead bias.
- Simulates stop/target/time exits.
- Saves results to `data/backtests/`.

`backtest_metrics.py`

- Calculates trades, win rate, average R, expectancy, profit factor, total return, max drawdown, holding period, exit counts.

`backtest_visuals.py`

- Builds equity curve and drawdown Plotly figures.

`backtest_ai_review.py`

- Optional Ollama review of backtest metrics and recent trades.

`strategy_presets.py`

- Saves/loads backtest strategy presets in `data/strategy_presets.json`.

### AI / Ollama Modules

`ollama_client.py`

- Checks Ollama availability.
- Lists local models.
- Calls Ollama for JSON or text.
- Parses JSON responses safely.

`ai_prompt_templates.py`

- Builds candidate review prompts.

`ai_validation.py`

- Validates/clamps AI review JSON.

`ai_review.py`

- Defines `AIReviewSettings`.
- Reviews strategy candidates.
- Caches AI reviews in `data/ai_reviews/`.
- Integrates AI score with deterministic score when enabled.

`ai_watchlist_curator.py`

- Optional AI weekly curator that groups candidates into watchlist categories.

Important: Ollama is optional. The app should work if Ollama is offline or returns invalid JSON.

### `journal.py`

New trade journal subsystem.

Creates and manages:

- `data/journal/planned_trades.csv`
- `data/journal/trades.csv`
- `data/journal/ai_plan_reviews.jsonl`
- `data/journal/ai_trade_reviews.jsonl`
- `data/journal/journal_lessons.jsonl`
- `data/journal/trade_scorecards.jsonl`
- `data/journal/trade_screenshots/`

Important functions:

- `ensure_journal_files()`
- `load_planned_trades()`
- `load_trades()`
- `create_planned_trade()`
- `validate_plan()`
- `calculate_plan_fields()`
- `actualize_trade()`
- `close_active_trade()`
- `build_trade_hindsight_packet()`
- `run_ai_plan_review()`
- `run_ai_trade_review()`
- `execution_score_formula()`
- `pnl_summary()`
- `grouped_performance()`
- `planned_vs_actual_table()`
- `journal_coach_summary()`

Trade lifecycle:

```text
Planned Trade
  -> Active Trade
  -> Closed Trade / Final Journal Entry
```

The user should never manually type `plan_id` or `trade_id`. These are generated automatically.

### `watchlist_manager.py`

Loads/saves watchlist and category CSV files under `data/`.

### `alpha_vantage.py`

Alpha Vantage helper module.

Current status:

- Loads API key from `.env`.
- Uses `ALPHA_VANTAGE_API_KEY`.
- Does not hardcode keys.
- Not heavily wired into the dashboard yet.

Future use should be sparse, such as confirming only top candidates.

### `main.py`

Scratch/test file only. Not the production Streamlit entry point.

## Local Data Files

Important generated/local data:

- `data/watchlist_master.csv`
- `data/categories.csv`
- `data/watchlist_market_cache.csv`
- `data/scans/`
- `data/backtests/`
- `data/ai_reviews/`
- `data/journal/`
- `data/strategy_presets.json`
- `data_cache/`
- `data/yfinance_cache/`

Treat these as local working data. Do not delete casually.

## Trade Journal Details

The Trade Journal page has these tabs:

1. Plan New Trade
2. Planned Trades
3. Active Trades
4. Closed Trade Journal
5. P&L Summary
6. AI Journal Coach
7. Personal Edge
8. Planned vs Actual Review

### Plan New Trade

Manual fields only:

- setup date
- ticker
- strategy
- setup type
- planned entry price
- planned stop loss
- planned exit / target price
- planned shares

Auto-calculated:

- plan ID
- status
- exposure
- risk per share
- total risk
- reward per share
- total reward
- risk/reward
- planned return %
- planned loss %
- max loss
- max gain
- AI plan score/confidence/feedback when AI review is run

### Planned Trades

Supports:

- edit
- run or rerun AI plan review
- actualize trade
- mark cancelled
- mark skipped
- delete with confirmation checkbox

### Active Trades

Actualized trades stay active until closed. Closing creates the final row in `trades.csv`; it does not delete the planned trade row.

### Closed Trades

Final record includes planned fields, active entry fields, close/reflection fields, P&L fields, R multiple, MFE/MAE, plan-vs-actual values, formula score, and optional AI grade.

### Planned vs Actual Review

Uses daily yfinance OHLCV to estimate:

- whether target or stop was hit during holding period
- MFE/MAE
- whether target or stop was hit after exit
- estimated plan-following P&L
- whether following the plan may have done better

If a daily candle touches both target and stop, intraday sequence is uncertain and should be treated as an estimate.

## Current App Status

Working milestone as of this session:

- App runs at `http://127.0.0.1:8502`.
- Core modules compile.
- Comprehensive Scanner page exists.
- Backtesting Lab exists.
- Trade Journal exists.
- Chart Viewer has simplified TradingView-style defaults.
- Persistent yfinance watchlist cache exists.
- Ollama integration is optional and should degrade gracefully.

Validation commands run during this session:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py journal.py scanner.py scan_logger.py backtester.py backtest_metrics.py backtest_visuals.py backtest_ai_review.py strategy_presets.py
```

Journal smoke checks confirmed:

- Empty journal files load without crashing.
- Planned trade calculations work.
- Close-trade calculations and formula score work on synthetic data.

## Known Limitations / Risks

- yfinance can be unreliable or blocked in restricted shell environments.
- Backtesting is daily-bar research only.
- Daily OHLCV cannot determine exact intraday order when stop and target are both touched.
- AI review quality depends on Ollama availability and JSON validity.
- Trade journal edit/delete flows are functional but can be polished further.
- Current backtester prioritizes correctness/readability over speed for large universes.
- Alpha Vantage confirmation remains a future enhancement.

## Current Git / Worktree Note

The worktree contains many uncommitted modified and new files from this session. Do not revert unrelated changes. Before a major new feature, consider making a local Git checkpoint.

Files added or heavily changed this session include:

- `app.py`
- `data_yfinance.py`
- `charting.py`
- `scanner.py`
- `scan_logger.py`
- `backtester.py`
- `backtest_metrics.py`
- `backtest_visuals.py`
- `backtest_ai_review.py`
- `strategy_presets.py`
- `journal.py`
- `ollama_client.py`
- `ai_review.py`
- `ai_validation.py`
- `ai_prompt_templates.py`
- `ai_watchlist_curator.py`
- `strategy_engine.py`
- `trade_plan.py`
- advanced technical modules listed above

## Recommended Next Steps

High-value next work:

1. Make a local Git checkpoint.
2. Add or update `.gitignore` for generated logs/cache/data if needed.
3. Add small automated tests for `journal.py` calculations.
4. Improve Trade Journal editing UX and confirmations.
5. Add export/import tools for journal data.
6. Improve backtest speed for larger ticker universes.
7. Add Alpha Vantage confirmation only for top candidates.
8. Continue refining strategy logic based on real journal results.

## Development Principle

Prioritize a working, testable app over perfect trading logic.

Build in small milestones. Keep changes modular. Preserve user data. Explain what changed and how to use it.

## Documentation Maintenance Rule

After any major project change, update all project context files so future VS Code / Codex / Copilot sessions have current information.

Update each file according to its scope:

- `README.md`: high-level project overview, current capabilities, limitations, current phase, next steps, and how to run.
- `PROJECT_CONTEXT.md`: detailed implementation context, architecture decisions, file/module status, workflow notes, safety rules, and handoff details.
- `.github/copilot-instructions.md`: concise repo-level instructions if repo-level workflows or safety rules change.
