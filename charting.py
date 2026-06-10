from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CACHE_DIR = Path("data_cache")


def add_chart_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA_100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()

    df["SMA_200"] = df["close"].rolling(200).mean()

    return df


def get_cached_tickers() -> list[str]:
    if not CACHE_DIR.exists():
        return []

    tickers = []

    for file in CACHE_DIR.glob("*.csv"):
        name = file.stem
        ticker = name.split("_")[0].upper()

        if ticker not in tickers:
            tickers.append(ticker)

    return sorted(tickers)


def calculate_support_resistance(
    df: pd.DataFrame, lookback: int = 60
) -> tuple[float, float]:
    recent = df.tail(lookback)

    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    return support, resistance


def create_trading_chart(
    df: pd.DataFrame,
    ticker: str,
    support_lookback: int = 60,
    show_ema_10: bool = True,
    show_ema_20: bool = True,
    show_ema_50: bool = True,
    show_ema_100: bool = True,
    show_ema_200: bool = True,
    show_sma_200: bool = True,
    show_support_resistance: bool = True,
):
    df = add_chart_indicators(df)

    support, resistance = calculate_support_resistance(df, support_lookback)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    if show_ema_10:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_10"], mode="lines", name="EMA 10"),
            row=1,
            col=1,
        )

    if show_ema_20:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_20"], mode="lines", name="EMA 20"),
            row=1,
            col=1,
        )

    if show_ema_50:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_50"], mode="lines", name="EMA 50"),
            row=1,
            col=1,
        )

    if show_ema_100:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_100"], mode="lines", name="EMA 100"),
            row=1,
            col=1,
        )

    if show_ema_200:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_200"], mode="lines", name="EMA 200"),
            row=1,
            col=1,
        )

    if show_sma_200:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_200"], mode="lines", name="SMA 200"),
            row=1,
            col=1,
        )

    if show_support_resistance:
        fig.add_hline(
            y=support,
            line_dash="dash",
            annotation_text=f"Support: {support:.2f}",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

        fig.add_hline(
            y=resistance,
            line_dash="dash",
            annotation_text=f"Resistance: {resistance:.2f}",
            annotation_position="top right",
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{ticker.upper()} Candlestick Chart",
        height=800,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig
