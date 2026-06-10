import pandas as pd

from indicators import add_indicators


def score_stock(ticker: str, price_data: pd.DataFrame) -> dict:
    df = add_indicators(price_data)

    latest = df.dropna().iloc[-1]

    score = 0
    reasons = []

    # Trend score
    if latest["close"] > latest["SMA_20"]:
        score += 20
        reasons.append("Close is above 20-day SMA")

    if latest["close"] > latest["SMA_50"]:
        score += 20
        reasons.append("Close is above 50-day SMA")

    if latest["SMA_20"] > latest["SMA_50"]:
        score += 15
        reasons.append("20-day SMA is above 50-day SMA")

    # Momentum score
    if latest["return_5d"] > 0:
        score += 10
        reasons.append("Positive 5-day momentum")

    if latest["return_20d"] > 0:
        score += 10
        reasons.append("Positive 20-day momentum")

    # RSI score
    if 45 <= latest["RSI_14"] <= 70:
        score += 15
        reasons.append("RSI is in a constructive range")
    elif latest["RSI_14"] > 75:
        score -= 10
        reasons.append("RSI may be overextended")

    # Volume confirmation
    if latest["volume_ratio"] > 1.2:
        score += 10
        reasons.append("Volume is above recent average")

    return {
        "ticker": ticker,
        "score": round(score, 2),
        "latest_close": round(float(latest["close"]), 2),
        "rsi": round(float(latest["RSI_14"]), 2),
        "return_5d_pct": round(float(latest["return_5d"] * 100), 2),
        "return_20d_pct": round(float(latest["return_20d"] * 100), 2),
        "volume_ratio": round(float(latest["volume_ratio"]), 2),
        "reasons": "; ".join(reasons),
    }


def rank_stocks(price_data_by_ticker: dict) -> pd.DataFrame:
    results = []

    for ticker, price_data in price_data_by_ticker.items():
        try:
            results.append(score_stock(ticker, price_data))
        except Exception as error:
            results.append(
                {
                    "ticker": ticker,
                    "score": 0,
                    "latest_close": None,
                    "rsi": None,
                    "return_5d_pct": None,
                    "return_20d_pct": None,
                    "volume_ratio": None,
                    "reasons": f"Error: {error}",
                }
            )

    ranking = pd.DataFrame(results)

    if not ranking.empty:
        ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)
        ranking.insert(0, "rank", range(1, len(ranking) + 1))

    return ranking
