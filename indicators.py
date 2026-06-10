import pandas as pd


def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    delta = close_prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

    df["RSI_14"] = calculate_rsi(df["close"], 14)

    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    df["volume_avg_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_avg_20"]

    return df
