# Copilot Instructions

This repo is a personal swing-trading research dashboard built with Python and Streamlit.

## User Context

The user is newer to Python, VS Code, virtual environments, Streamlit, Git, and project structure.

When suggesting steps:

- Be beginner-friendly and explicit.
- Say whether commands run in PowerShell or code goes inside a `.py` file.
- Do not assume the user knows the difference between PowerShell and the Python `>>>` shell.
- Prefer small, working milestones over large rewrites.

## Safety Rules

- Do not delete files unless the user explicitly asks.
- Do not hardcode API keys.
- Do not expose `.env` contents.
- Ask before installing packages, resetting Git, deleting files, or making large changes.
- This is for personal research and education, not guaranteed trading predictions.
- Avoid direct "buy now" or "sell now" certainty. Keep analysis cautious and risk-aware.

## Project Architecture

`app.py` is the main Streamlit dashboard and should remain the main entry point.

The intended flow is:

```text
app.py
  -> data_yfinance.py pulls OHLCV data
  -> indicators.py calculates indicators
  -> ranking_model.py scores/ranks stocks
  -> local_ai_review.py sends ranking to Ollama
  -> app.py displays results
```

Alpha Vantage should be secondary for now. Use yfinance for development and bulk watchlist scans. Later, Alpha Vantage should only confirm the top 3-5 ranked candidates, not scan the whole watchlist.

## Current Main Files

- `app.py`: main Streamlit app with Single Stock Analysis, Watchlist Ranking, and Chart Viewer.
- `data_yfinance.py`: yfinance download/cache helpers.
- `indicators.py`: RSI, moving averages, returns, volume ratio.
- `ranking_model.py`: starter scoring/ranking model.
- `charting.py`: Plotly candlestick chart with volume, EMA/SMA overlays, and simple support/resistance.
- `alpha_vantage.py`: Alpha Vantage helpers using `.env`.
- `local_ai_review.py`: Ollama helper using `llama3.2`.
- `PROJECT_CONTEXT.md`: longer project memory and handoff notes.

## Common PowerShell Commands

Run from the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
uv pip install package_name
uv pip freeze > requirements.txt
git status
git log --oneline
```

Stop Streamlit with `Ctrl + C` in the terminal.

## Development Priorities

Build incrementally:

1. Keep the current app working.
2. Preserve working milestones with Git before larger changes.
3. Prefer yfinance and cached data for early testing.
4. Improve ranking table cleanup with a real `rank` column.
5. Add AI review only after ranking works reliably.
6. Add Alpha Vantage confirmation only for top candidates.
7. Add trade journal and P&L dashboard later.

## Ranking Cleanup To Preserve

When sorting ranking results, prefer:

```python
ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)
ranking.insert(0, "rank", range(1, len(ranking) + 1))
```

In Streamlit tables, prefer:

```python
st.dataframe(ranking, use_container_width=True, hide_index=True)
```
