# Swing Trading Model

This is a personal swing-trading research dashboard built with Python and Streamlit. The goal is to help scan stocks, calculate technical indicators, rank possible setups, and review candidates in a more organized, risk-aware way.

This project is for research and education. It is not meant to provide guaranteed predictions or automatic buy/sell decisions.

## What The App Does

The app currently uses yfinance as the main data source for daily stock price and volume data. It calculates technical indicators locally, scores stocks with a starter ranking model, and displays results in a Streamlit dashboard.

Current dashboard pages:

- Single Stock Analysis: enter one ticker and view recent price data, RSI, returns, moving averages, and volume.
- Watchlist Ranking: enter multiple tickers and rank them using a simple technical scoring model.
- Chart Viewer: view a Plotly candlestick chart with volume, moving averages, and basic support/resistance lines.

## Current Model Logic

The current ranking model is a starter proof of concept. It scores stocks using simple technical rules such as:

- Price above short-term moving averages
- Positive 5-day and 20-day momentum
- RSI in a constructive range
- Volume above recent average

The scoring logic is intentionally simple for now so the dashboard can be tested and improved step by step.

## Limitations

Current limitations:

- The ranking model is not a final trading strategy.
- It does not yet account for earnings dates, news risk, sector strength, market regime, stop-loss distance, or portfolio risk.
- yfinance is useful for development, but it may not always be perfectly reliable.
- Alpha Vantage is available as a helper module, but it is not yet used for top-candidate confirmation in the dashboard.
- Local AI review through Ollama exists as a helper file, but it is not wired into the dashboard yet.
- The app does not currently track trades, P&L, win rate, drawdown, or journal notes.

## Current Phase

The project is currently in the early dashboard-building phase.

Completed or mostly working:

- Streamlit app shell
- yfinance data loading
- Local CSV caching
- Basic indicator calculations
- Watchlist ranking
- Basic candlestick chart viewer

Immediate next steps:

- Make a Git checkpoint of the current working state.
- Clean up the ranking table with a real rank column and hidden pandas index.
- Confirm the app runs reliably from `app.py`.
- Continue improving the Chart Viewer.
- Add local AI review after ranking is stable.
- Add Alpha Vantage confirmation only for the top few ranked candidates.

## Documentation Maintenance

After major project changes, update the project docs so future sessions stay aligned:

- Update this `README.md` with the high-level app status, limitations, current phase, and next steps.
- Update `PROJECT_CONTEXT.md` with the detailed implementation context and handoff notes.
- Update `.github/copilot-instructions.md` if repo-level guidance, workflows, or safety rules change.

## Project Roadmap

Planned phases:

1. Build a basic Streamlit app for single-stock analysis.
2. Add yfinance data loading and local caching.
3. Add local technical indicators.
4. Add a starter stock ranking model.
5. Add a chart viewer for candlesticks, volume, moving averages, and support/resistance.
6. Add local AI review using Ollama.
7. Add Alpha Vantage confirmation for only the top 3-5 candidates.
8. Add a trade journal.
9. Add a P&L dashboard.
10. Improve the scoring model with risk, market regime, relative strength, volatility, event risk, and portfolio constraints.

## How To Run

From PowerShell in the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Then open the Streamlit URL shown in the terminal.
