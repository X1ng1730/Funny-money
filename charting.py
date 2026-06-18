from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from support_resistance import calculate_support_resistance_levels

CACHE_DIR = Path("data_cache")
EMA_8_COLOR = "#c7a7ff"
SMA_200_COLOR = "#ff9fcb"
LEVEL_COLOR = "rgba(255,255,255,0.88)"
SWING_LEVEL_COLOR = "rgba(255,255,255,0.55)"
TAKE_PROFIT_FILL = "rgba(29, 185, 84, 0.16)"
SELL_ZONE_FILL = "rgba(255, 75, 92, 0.14)"


def add_chart_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["EMA_8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["EMA_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA_100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()

    df["SMA_50"] = df["close"].rolling(50).mean()
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

    if recent.empty:
        raise ValueError("Not enough price data to calculate support and resistance")

    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    if pd.isna(support) or pd.isna(resistance):
        raise ValueError("Support and resistance could not be calculated")

    return support, resistance


def _nearest_pivot_levels(levels: dict, price: float | None) -> tuple[float | None, float | None]:
    if price is None:
        return None, None

    pivot_lows = [level for level in levels.get("pivot_lows", []) if level is not None and level <= price]
    pivot_highs = [level for level in levels.get("pivot_highs", []) if level is not None and level >= price]
    swing_low = max(pivot_lows) if pivot_lows else levels.get("nearest_support")
    swing_high = min(pivot_highs) if pivot_highs else levels.get("nearest_resistance")
    return swing_low, swing_high


def _add_clean_level(fig, y: float | None, text: str, dash: str = "dash", color: str = LEVEL_COLOR, position: str = "top right") -> None:
    if y is None or pd.isna(y):
        return
    fig.add_hline(
        y=y,
        line_dash=dash,
        line_color=color,
        line_width=1.4,
        annotation_text=f"{text}: {y:.2f}",
        annotation_position=position,
        annotation_font_color="white",
        annotation_font_size=11,
        row=1,
        col=1,
    )


def create_trading_chart(
    df: pd.DataFrame,
    ticker: str,
    support_lookback: int = 60,
    show_ema_10: bool = False,
    show_ema_8: bool = True,
    show_ema_9: bool = False,
    show_ema_20: bool = False,
    show_ema_21: bool = False,
    show_ema_50: bool = False,
    show_ema_100: bool = False,
    show_ema_200: bool = False,
    show_sma_50: bool = False,
    show_sma_200: bool = True,
    show_support_resistance: bool = True,
    show_volume_profile: bool = False,
    show_fvgs: bool = False,
    show_order_blocks: bool = False,
    show_vwap: bool = False,
    show_liquidity_sweeps: bool = False,
    show_strategy_zones: bool = True,
    trade_plan: dict | None = None,
):
    if df.empty:
        raise ValueError("No price data available to chart")

    df = add_chart_indicators(df)

    support, resistance = calculate_support_resistance(df, support_lookback)
    levels = calculate_support_resistance_levels(df)
    latest_price = float(df["close"].dropna().iloc[-1]) if not df["close"].dropna().empty else None
    swing_low, swing_high = _nearest_pivot_levels(levels, latest_price)
    atr_proxy = float((df["high"] - df["low"]).tail(14).mean()) if len(df) >= 14 else None

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
            increasing_line_color="#00b894",
            increasing_fillcolor="#00b894",
            decreasing_line_color="#ff4b5c",
            decreasing_fillcolor="#ff4b5c",
        ),
        row=1,
        col=1,
    )

    if show_ema_8:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_8"],
                mode="lines",
                name="8 EMA",
                line=dict(color=EMA_8_COLOR, width=2.2),
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

    if show_ema_9:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_9"], mode="lines", name="EMA 9"),
            row=1,
            col=1,
        )

    if show_ema_20:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_20"], mode="lines", name="EMA 20"),
            row=1,
            col=1,
        )

    if show_ema_21:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA_21"], mode="lines", name="EMA 21"),
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

    if show_sma_50:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_50"], mode="lines", name="SMA 50"),
            row=1,
            col=1,
        )

    if show_sma_200:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_200"],
                mode="lines",
                name="200 SMA",
                line=dict(color=SMA_200_COLOR, width=2.4),
            ),
            row=1,
            col=1,
        )

    if show_support_resistance:
        _add_clean_level(fig, resistance, "Overall resistance", "dash", LEVEL_COLOR, "top right")
        _add_clean_level(fig, support, "Overall support", "dash", LEVEL_COLOR, "bottom right")
        _add_clean_level(fig, swing_high, "Swing-high resistance", "dot", SWING_LEVEL_COLOR, "top left")
        _add_clean_level(fig, swing_low, "Swing-low support", "dot", SWING_LEVEL_COLOR, "bottom left")

    if show_strategy_zones:
        target_1 = trade_plan.get("target_1") if trade_plan else None
        target_2 = trade_plan.get("target_2") if trade_plan else None
        stop_price = trade_plan.get("stop_price") if trade_plan else None
        entry_low = trade_plan.get("entry_price_low") if trade_plan else None
        entry_high = trade_plan.get("entry_price_high") if trade_plan else None

        take_profit_low = target_1 or swing_high or resistance
        take_profit_high = target_2 or (take_profit_low + (atr_proxy or 0) * 0.75 if take_profit_low is not None else None)
        if take_profit_low is not None and take_profit_high is not None and take_profit_high >= take_profit_low:
            fig.add_hrect(
                y0=take_profit_low,
                y1=take_profit_high,
                fillcolor=TAKE_PROFIT_FILL,
                line_width=0,
                annotation_text="Take-profit zone",
                annotation_position="top left",
                annotation_font_color="white",
                row=1,
                col=1,
            )

        sell_zone_high = stop_price or swing_low or support
        sell_zone_low = sell_zone_high - (atr_proxy or max(sell_zone_high * 0.015, 0.01)) if sell_zone_high is not None else None
        if sell_zone_low is not None and sell_zone_high is not None:
            fig.add_hrect(
                y0=sell_zone_low,
                y1=sell_zone_high,
                fillcolor=SELL_ZONE_FILL,
                line_width=0,
                annotation_text="Sell / invalidation zone",
                annotation_position="bottom left",
                annotation_font_color="white",
                row=1,
                col=1,
            )

        if entry_low is not None and entry_high is not None:
            fig.add_hrect(
                y0=entry_low,
                y1=entry_high,
                fillcolor="rgba(199, 167, 255, 0.10)",
                line_width=0,
                annotation_text="Entry zone",
                annotation_position="top right",
                annotation_font_color="white",
                row=1,
                col=1,
            )

    if trade_plan and show_volume_profile:
        for label, key, color in [
            ("POC proxy", "vp_poc", "rgba(241, 196, 15, 0.35)"),
            ("Value high", "vp_value_area_high", "rgba(52, 152, 219, 0.25)"),
            ("Value low", "vp_value_area_low", "rgba(52, 152, 219, 0.25)"),
            ("Nearest HVN", "vp_nearest_hvn_above", "rgba(155, 89, 182, 0.25)"),
        ]:
            level = trade_plan.get(key)
            if level is not None and pd.notna(level):
                fig.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color=color,
                    annotation_text=f"{label}: {level:.2f}",
                    annotation_position="top left",
                    row=1,
                    col=1,
                )

    if trade_plan and show_fvgs:
        fvg_low = trade_plan.get("bullish_fvg_low")
        fvg_high = trade_plan.get("bullish_fvg_high")
        if fvg_low is not None and fvg_high is not None:
            fig.add_hrect(
                y0=fvg_low,
                y1=fvg_high,
                fillcolor="rgba(46, 204, 113, 0.10)",
                line_width=0,
                annotation_text="Estimated bullish FVG",
                row=1,
                col=1,
            )

    if trade_plan and show_order_blocks:
        ob_low = trade_plan.get("bullish_order_block_low")
        ob_high = trade_plan.get("bullish_order_block_high")
        if ob_low is not None and ob_high is not None:
            fig.add_hrect(
                y0=ob_low,
                y1=ob_high,
                fillcolor="rgba(230, 126, 34, 0.12)",
                line_width=0,
                annotation_text="Estimated bullish order block",
                row=1,
                col=1,
            )

    if trade_plan and show_vwap:
        vwap_level = trade_plan.get("intraday_vwap")
        if vwap_level is not None and pd.notna(vwap_level):
            fig.add_hline(
                y=vwap_level,
                line_dash="dashdot",
                line_color="purple",
                annotation_text=f"VWAP proxy: {vwap_level:.2f}",
                annotation_position="bottom left",
                row=1,
                col=1,
            )

    if trade_plan and show_liquidity_sweeps:
        swept = trade_plan.get("swept_level")
        if swept is not None and pd.notna(swept):
            fig.add_hline(
                y=swept,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Swept level: {swept:.2f}",
                annotation_position="bottom right",
                row=1,
                col=1,
            )

    if trade_plan:
        entry_low = trade_plan.get("entry_price_low")
        entry_high = trade_plan.get("entry_price_high")
        stop_price = trade_plan.get("stop_price")
        targets = [
            ("Target 1", trade_plan.get("target_1")),
            ("Target 2", trade_plan.get("target_2")),
            ("Target 3", trade_plan.get("target_3")),
        ]
        if stop_price is not None:
            fig.add_hline(
                y=stop_price,
                line_dash="dash",
                line_color="#ff4b5c",
                line_width=1.4,
                annotation_text=f"Stop: {stop_price:.2f}",
                annotation_position="bottom left",
                annotation_font_color="white",
                row=1,
                col=1,
            )
        for label, target in targets:
            if target is not None and pd.notna(target):
                fig.add_hline(
                    y=target,
                    line_dash="dot",
                    line_color="#1db954",
                    line_width=1.3,
                    annotation_text=f"{label}: {target:.2f}",
                    annotation_position="top right",
                    annotation_font_color="white",
                    row=1,
                    col=1,
                )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker_color="#f7c85f",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{ticker.upper()} Candlestick Chart",
        height=800,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#f5f7fb"),
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
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.10)")

    return fig
