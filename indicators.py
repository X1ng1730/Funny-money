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


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["EMA_8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["EMA_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["SMA_200"] = df["close"].rolling(200).mean()

    df["RSI_14"] = calculate_rsi(df["close"], 14)
    df["ATR_14"] = calculate_atr(df, 14)
    df["ATR_pct"] = df["ATR_14"] / df["close"] * 100

    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)
    df["return_1m"] = df["return_20d"]
    df["return_3m"] = df["close"].pct_change(63)
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    df["close_vs_previous_close_pct"] = df["return_1d"] * 100

    df["volume_avg_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_avg_20"]
    df["relative_volume"] = df["volume_ratio"]
    df["dollar_volume_avg_20"] = df["close"] * df["volume_avg_20"]
    df["high_52w"] = df["high"].rolling(252, min_periods=20).max()
    df["low_52w"] = df["low"].rolling(252, min_periods=20).min()
    df["high_20d"] = df["high"].rolling(20, min_periods=5).max()
    df["high_50d"] = df["high"].rolling(50, min_periods=10).max()
    df["high_100d"] = df["high"].rolling(100, min_periods=20).max()
    df["low_20d"] = df["low"].rolling(20, min_periods=5).min()
    df["low_50d"] = df["low"].rolling(50, min_periods=10).min()
    df["low_100d"] = df["low"].rolling(100, min_periods=20).min()
    df["distance_from_8ema_pct"] = (df["close"] - df["EMA_8"]) / df["close"] * 100
    df["distance_from_9ema_pct"] = (df["close"] - df["EMA_9"]) / df["close"] * 100
    df["distance_from_21ema_pct"] = (df["close"] - df["EMA_21"]) / df["close"] * 100
    df["distance_from_50sma_pct"] = (df["close"] - df["SMA_50"]) / df["close"] * 100
    df["distance_from_200sma_pct"] = (df["close"] - df["SMA_200"]) / df["close"] * 100

    return df


def latest_complete_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("No price history available")

    preferred = df.dropna(subset=["close"])
    if preferred.empty:
        raise ValueError("No closing prices available")
    return preferred.iloc[-1]
