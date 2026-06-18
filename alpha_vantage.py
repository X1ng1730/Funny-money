import os
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://www.alphavantage.co/query"


def alpha_vantage_request(function: str, **params: Any) -> dict[str, Any]:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        raise RuntimeError("Missing ALPHA_VANTAGE_API_KEY. Add it to your .env file.")

    request_params = {
        "function": function,
        "apikey": api_key,
        **params,
    }

    response = requests.get(BASE_URL, params=request_params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")

    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate-limit message: {data['Note']}")

    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage message: {data['Information']}")

    return data


def get_daily_prices(symbol: str, outputsize: str = "compact") -> pd.DataFrame:
    data = alpha_vantage_request(
        "TIME_SERIES_DAILY",
        symbol=symbol.upper(),
        outputsize=outputsize,
    )

    time_series_key = "Time Series (Daily)"

    if time_series_key not in data:
        raise ValueError(f"Could not find daily price data for {symbol}: {data}")

    df = pd.DataFrame.from_dict(data[time_series_key], orient="index")

    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
    )

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    numeric_columns = ["open", "high", "low", "close", "volume"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

    return df


def get_global_quote(symbol: str) -> dict[str, Any]:
    data = alpha_vantage_request("GLOBAL_QUOTE", symbol=symbol.upper())
    return data.get("Global Quote", {})


def get_rsi(symbol: str, time_period: int = 14) -> pd.DataFrame:
    data = alpha_vantage_request(
        "RSI",
        symbol=symbol.upper(),
        interval="daily",
        time_period=time_period,
        series_type="close",
    )

    rsi_key = "Technical Analysis: RSI"

    if rsi_key not in data:
        raise ValueError(f"Could not find RSI data for {symbol}: {data}")

    df = pd.DataFrame.from_dict(data[rsi_key], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df["RSI"] = pd.to_numeric(df["RSI"])

    return df
