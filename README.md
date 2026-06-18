# Swing Trading Model

This is a personal stock swing-trading research dashboard built with Python and Streamlit. It helps organize watchlists, fetch daily market data, rank setups, review candidates with local Ollama AI, build trade plans, run strategy scans, backtest strategy rules, and keep a structured trade journal.

This project is for personal research and education. It does not provide guaranteed predictions, brokerage execution, automated trading, short selling, or options trading.

## Current App Capabilities

The app currently uses yfinance as the main market data source and keeps local caches so repeated app refreshes are faster. Alpha Vantage remains available as a helper module, but the main workflow is yfinance-first.

Current Streamlit pages:

- Single Stock Analysis: inspect one ticker with recent OHLCV data and indicators.
- Watchlist Ranking: rank multiple tickers with the starter technical model.
- Watchlist Dashboard: manage active watchlist rows and cached market snapshots.
- Strategy Scanner: quick filtered view of strategy matches.
- Comprehensive Scanner: manually run a full watchlist scan, apply filters, optionally run Ollama AI review, and save scan results.
- Weekly Trade Watchlist: generate a focused watchlist from the strongest strategy candidates.
- Backtesting Lab: test the same strategy engine historically with next-day entries, metrics, equity/drawdown visuals, presets, parameter sweeps, and optional AI review.
- Trade Journal: manage the full trade lifecycle from planned trade to active trade to closed journal entry.
- Chart Viewer: view a cleaner dark-mode candlestick chart with 8 EMA, 200 SMA, white support/resistance lines, and trade plan zones.

## Major Improvements Added This Session

- Added persistent watchlist market cache in `data_yfinance.py` to reduce slow reloads from repeated yfinance calls.
- Simplified the main chart so it focuses on 8 EMA, 200 SMA, support/resistance, take-profit zones, and sell/invalidation zones.
- Added strategy-aware scanner modules:
  - `scanner.py`
  - `scan_logger.py`
- Added Backtesting Lab modules:
  - `backtester.py`
  - `backtest_metrics.py`
  - `backtest_visuals.py`
  - `backtest_ai_review.py`
  - `strategy_presets.py`
- Added local Ollama AI workflow modules:
  - `ollama_client.py`
  - `ai_review.py`
  - `ai_validation.py`
  - `ai_prompt_templates.py`
  - `ai_watchlist_curator.py`
- Added advanced strategy/technical support modules:
  - `strategy_engine.py`
  - `trade_plan.py`
  - `support_resistance.py`
  - `volume_profile.py`
  - `acceptance_rejection.py`
  - `liquidity.py`
  - `fair_value_gap.py`
  - `order_blocks.py`
  - `vwap.py`
  - `confluence.py`
- Added a full trade journal subsystem in `journal.py` with planned trades, active trades, closed trades, P&L summary, AI hindsight grading, planned-vs-actual review, and personal edge tables.

## Trade Journal Workflow

The Trade Journal now follows a clear lifecycle:

```text
Plan trade
  -> optional AI pre-trade review
  -> actualize into active trade
  -> close trade
  -> calculate P&L, R multiple, MFE/MAE, plan-vs-actual differences
  -> optional AI hindsight grade
  -> journal lessons and personal edge review
```

Journal data is stored locally in:

- `data/journal/planned_trades.csv`
- `data/journal/trades.csv`
- `data/journal/ai_plan_reviews.jsonl`
- `data/journal/ai_trade_reviews.jsonl`
- `data/journal/journal_lessons.jsonl`
- `data/journal/trade_scorecards.jsonl`
- `data/journal/trade_screenshots/`

The planned trade form intentionally stays simple. It only asks for setup date, ticker, strategy, setup type, planned entry, planned stop, planned target, and planned shares. IDs, risk/reward, exposure, max loss, max gain, AI review fields, and journal context are generated automatically.

## Model And Strategy Logic

There are now two levels of scoring:

- `ranking_model.py`: a simple starter ranking model for basic watchlist sorting.
- `strategy_engine.py`: the main strategy evaluation layer used by scanner, weekly watchlist, chart context, and backtesting.

Current strategy families:

- Catalyst Gap / Multi-Month Breakout
- EMA Pullback Trend Continuation
- Reversal / Reclaim Setup

The strategy engine combines deterministic setup logic, advanced technical confluence, higher-timeframe context, risk flags, and generated trade plan levels.

## Ollama AI Integration

Ollama is optional. The app works without it.

When enabled, Ollama can:

- Review scanner candidates.
- Help curate the weekly watchlist.
- Review planned trades before entry.
- Grade closed trades in hindsight based on process, discipline, risk management, execution, and plan adherence.
- Summarize journal patterns through AI Journal Coach.
- Review backtest results.

The AI should use educational watchlist language and should not say that a user must buy or sell.

## Local Data And Generated Output

The app writes local files under:

- `data/`
- `data_cache/`

Important generated folders/files include:

- `data/watchlist_master.csv`
- `data/categories.csv`
- `data/watchlist_market_cache.csv`
- `data/scans/`
- `data/backtests/`
- `data/ai_reviews/`
- `data/journal/`
- `data/strategy_presets.json`

These are local working data files. Be careful before deleting or overwriting them.

## Current Limitations

- yfinance can be slow or temporarily unavailable, especially from restricted shell environments.
- Daily OHLCV hindsight cannot know exact intraday sequence if a candle hits both stop and target.
- Backtesting is daily-bar based and intended for research, not proof of future performance.
- The AI review depends on local Ollama availability and valid JSON responses.
- Alpha Vantage is still not part of the main candidate confirmation workflow.
- No brokerage integration, live execution, automated trading, options, or shorts.

## How To Run

From PowerShell in the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Then open:

```text
http://127.0.0.1:8502
```

If the app behaves like old code after edits, stop and restart Streamlit. A browser refresh alone may not reload changed Python modules.

## Useful Validation Commands

Compile the main app and newer modules:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py journal.py scanner.py scan_logger.py backtester.py backtest_metrics.py backtest_visuals.py backtest_ai_review.py strategy_presets.py
```

Check git status:

```powershell
git status --short
```

## Next Useful Improvements

- Add `.gitignore` coverage for generated local data/log/cache files if not already ignored.
- Add safer edit/delete confirmations for journal records beyond the current checkbox flow.
- Add journal import/export tools.
- Improve backtest speed for larger universes.
- Add Alpha Vantage confirmation only for top candidates.
- Add more robust automated tests around journal calculations and strategy backtests.

## Documentation Maintenance

After major project changes, update:

- `README.md`: high-level current state and how to run.
- `PROJECT_CONTEXT.md`: detailed implementation context and handoff notes.
- `.github/copilot-instructions.md` if repo-level workflows or safety rules change.
