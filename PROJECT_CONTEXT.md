# Swing Trading Model - Project Context

This file is a handoff note for future VS Code / Codex / Copilot chats. It summarizes the current state of the project and the intended development direction.

## Project Location

`C:\Users\xingy\OneDrive\Desktop\Swing Trading Model`

## User Experience Level

The user is still new to Python, VS Code, virtual environments, Streamlit, Git, Codex/Copilot, and project structure.

When giving instructions:

- Explain steps clearly.
- Specify whether commands should be run in PowerShell or written inside a `.py` file.
- Do not assume the user knows whether they are in PowerShell versus the Python `>>>` shell.
- Work incrementally and preserve working milestones.

## Safety Rules

- Do not delete files unless the user explicitly asks.
- Do not hardcode API keys.
- Do not read or expose `.env` contents unless absolutely necessary and explicitly approved.
- Ask before installing packages.
- Ask before deleting files.
- Ask before resetting Git.
- Ask before making large changes.
- Do not present dashboard output as guaranteed predictions or direct buy/sell instructions.
- Keep analysis risk-aware, cautious, and educational.

## Current Tech Stack

- Windows
- VS Code
- Python virtual environment: `.venv`
- Python installed through `uv`
- Package installation through `uv pip install ...`
- Streamlit for dashboard UI
- yfinance as the main development/testing data source
- Alpha Vantage as a secondary/cross-check source
- Ollama with `llama3.2` for local AI review
- Git for local checkpoints

## Important Commands

Run these in PowerShell from the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the Streamlit app:

```powershell
streamlit run app.py --server.port 8502
```

Safer Streamlit run command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Stop Streamlit:

```text
Ctrl + C
```

Install packages:

```powershell
uv pip install package_name
```

Update requirements:

```powershell
uv pip freeze > requirements.txt
```

Check Git:

```powershell
git status
git log --oneline
```

## Current Project Goal

Build a personal swing-trading research dashboard for stocks.

The target holding period is roughly 1-6 weeks. The dashboard should eventually scan a stock universe such as Nasdaq 100, S&P 500, and a custom watchlist.

Risk preferences discussed earlier:

- Max portfolio risk per trade: roughly 0.5%-2%
- Max total open risk: roughly 5%-8%
- Max open positions: roughly 3-5
- No margin
- Whole shares only

The long-term benchmark should be better than a basket/average of large ETFs such as SPY, QQQ, VOO, etc.

This is for personal research and decision support, not publishing.

## High-Level Target Flow

```text
app.py
  -> data_yfinance.py pulls OHLCV data
  -> indicators.py calculates indicators
  -> ranking_model.py scores/ranks stocks
  -> local_ai_review.py sends ranking to Ollama
  -> app.py displays everything in Streamlit
```

Later Alpha Vantage flow:

```text
yfinance scans 100+ stocks
  -> Python calculates indicators locally
  -> ranking_model.py ranks them
  -> Alpha Vantage checks only top 3-5 for earnings/news/fundamentals
  -> Ollama reviews final candidates
  -> Streamlit displays results
```

## Why yfinance Is Primary For Now

Alpha Vantage free tier is too limited for early development/testing. It can hit rate limits quickly, especially if each button click makes multiple requests.

During development, yfinance should be the main source for bulk price/volume data. Alpha Vantage should be used sparingly later, mainly for the top 3-5 ranked candidates.

## Current Files

### `app.py`

Main Streamlit dashboard / entry point.

Current pages:

- Single Stock Analysis
- Watchlist Ranking
- Chart Viewer

The final app should not require running every `.py` file separately. The main command should remain:

```powershell
streamlit run app.py --server.port 8502
```

### `data_yfinance.py`

Downloads OHLCV data from yfinance.

Current behavior:

- Uses `data_cache/`
- Provides `get_price_data()`
- Provides `get_multiple_price_data()`
- Normalizes yfinance columns to lowercase:
  - `open`
  - `high`
  - `low`
  - `close`
  - `volume`

### `indicators.py`

Calculates local indicators.

Current indicators:

- RSI 14
- SMA 20
- SMA 50
- EMA 20
- 5-day return
- 20-day return
- 20-day average volume
- Volume ratio

Future indicators:

- MACD
- ATR
- EMA 10/50/100/200 shared with the main indicator module
- Relative strength versus SPY/QQQ
- Gap behavior
- Volatility filters

### `ranking_model.py`

Scores and ranks stocks.

Current starter scoring:

- +20 if close is above 20-day SMA
- +20 if close is above 50-day SMA
- +15 if 20-day SMA is above 50-day SMA
- +10 if 5-day return is positive
- +10 if 20-day return is positive
- +15 if RSI is between 45 and 70
- -10 if RSI is above 75
- +10 if volume is above 1.2x 20-day average

Current output columns:

- `ticker`
- `score`
- `latest_close`
- `rsi`
- `return_5d_pct`
- `return_20d_pct`
- `volume_ratio`
- `reasons`

Known small cleanup needed:

- After sorting, reset the DataFrame index.
- Add a real `rank` column.
- Display tables with `hide_index=True`.

Suggested ranking cleanup:

```python
ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)
ranking.insert(0, "rank", range(1, len(ranking) + 1))
```

### `charting.py`

Creates Plotly chart figures.

Current chart features:

- Candlestick chart
- Volume subplot
- EMA 10
- EMA 20
- EMA 50
- EMA 100
- EMA 200
- SMA 200
- Simple support/resistance lines
- Cached ticker detection from `data_cache/`

Current support/resistance logic:

- Support = lowest low over selected lookback window
- Resistance = highest high over selected lookback window

Future improvements:

- Support/resistance zones
- Swing highs/lows
- Volume profile
- Trendlines
- Earnings markers
- Buy/sell markers
- RSI/MACD lower panels

### `alpha_vantage.py`

Alpha Vantage helper module.

Current behavior:

- Loads API key from `.env`
- Uses `ALPHA_VANTAGE_API_KEY`
- Does not hardcode API keys
- Provides:
  - `alpha_vantage_request()`
  - `get_daily_prices()`
  - `get_global_quote()`
  - `get_rsi()`

For now, Alpha Vantage should not be used heavily in `app.py`.

Later add a button such as:

```text
Confirm top 5 with Alpha Vantage
```

That button should only call Alpha Vantage for top ranked candidates, not the whole watchlist.

### `local_ai_review.py`

Ollama/local AI helper.

Current behavior:

- Uses the `ollama` Python package
- Uses model `llama3.2`
- Sends ranked stock text to a local model
- Asks for cautious, risk-aware review
- Avoids guaranteed predictions and direct buy/sell instructions

This file exists but is not wired into `app.py` yet.

### `main.py`

Scratch/test file only.

Currently contains a small yfinance test. It is not the final dashboard entry point.

### `requirements.txt`

Generated package list. Includes major dependencies such as:

- streamlit
- pandas
- yfinance
- plotly
- ollama
- python-dotenv
- requests

### `.env`

Private file for API keys.

Expected key:

```text
ALPHA_VANTAGE_API_KEY=...
```

Do not commit this file.

### `.gitignore`

Currently ignores:

- `__pycache__/`
- `*.pyc`
- `.venv/`
- `.env`
- `.ipynb_checkpoints/`
- `.vscode/mcp.json`

Recommended addition:

```text
data_cache/
```

### `data_cache/`

Local cached yfinance CSV files.

Current cached tickers include examples such as:

- AAPL
- AMD
- AMZN
- GOOGL
- META
- MSFT
- NVDA
- TSLA

This folder should usually be ignored by Git unless intentionally saving sample data.

## Current Git Status Observed

At the time this context file was created, `git status --short` showed uncommitted changes/new files including:

- Modified `app.py`
- Modified `requirements.txt`
- New `charting.py`
- New `data_yfinance.py`
- New `indicators.py`
- New `ranking_model.py`
- New `vscode-extensions.txt`
- New `data_cache/`

Before larger work, consider making a local Git checkpoint.

## Current Dashboard Status

The Streamlit app appears to be in a working milestone state.

It already has:

- yfinance as primary data source
- Local data caching
- Single stock analysis
- Watchlist ranking
- Basic chart viewer
- Plotly candlestick chart
- Moving average overlays
- Simple support/resistance

It does not yet have:

- AI review wired into the dashboard
- Alpha Vantage confirmation button for top candidates
- Trade journal
- P&L dashboard
- Advanced scoring/risk model

## Recommended Safest Next Step

Before making more functional changes, preserve the current working milestone with Git.

Then make the smallest useful cleanup:

1. Add `data_cache/` to `.gitignore`.
2. Update `ranking_model.py` to reset the sorted index and add a real `rank` column.

This is low-risk, matches the project notes, and improves the Streamlit ranking table without changing the whole system.

## Phased Build Order

1. Confirm app runs.
2. Confirm yfinance cache works.
3. Confirm single stock chart works.
4. Confirm watchlist ranking works.
5. Confirm rank column/index cleanup.
6. Add or polish Chart Viewer.
7. Add AI review using Ollama.
8. Add Alpha Vantage confirmation for top candidates.
9. Add trade journal.
10. Add P&L dashboard.
11. Improve scoring/risk model.

## Development Principle

Prioritize a working, testable app over perfect trading logic.

Build in small milestones. Before major edits, summarize what files will change and why.

## Documentation Maintenance Rule

After any major project change, update all project context files so future VS Code / Codex / Copilot sessions have current information.

Update each file according to its scope:

- `README.md`: high-level project overview, current app capabilities, limitations, current phase, next steps, and roadmap.
- `PROJECT_CONTEXT.md`: detailed implementation context, file/module status, architecture decisions, workflow notes, safety rules, and handoff details.
- `.github/copilot-instructions.md`: concise repo-level instructions for Copilot, including coding style, safety rules, project architecture, and current development priorities.

If a future assistant reads any one of these files and is asked to update project docs, it should update all three files when relevant.
