from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


def get_price_data(
    ticker: str, period: str = "6mo", interval: str = "1d", use_cache: bool = True
) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    cache_file = CACHE_DIR / f"{ticker}_{period}_{interval}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # yfinance sometimes returns multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df = df[["open", "high", "low", "close", "volume"]].dropna()

    df.to_csv(cache_file)

    return df


def get_multiple_price_data(
    tickers: list[str],
    period: str = "6mo",
    interval: str = "1d",
    use_cache: bool = True,
) -> dict:
    results = {}

    for ticker in tickers:
        try:
            results[ticker] = get_price_data(ticker, period, interval, use_cache)
        except Exception as error:
            print(f"Could not load {ticker}: {error}")

    return results
