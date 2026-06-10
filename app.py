import streamlit as st

from data_yfinance import get_multiple_price_data, get_price_data
from indicators import add_indicators
from ranking_model import rank_stocks
from charting import create_trading_chart, get_cached_tickers

st.title("Swing Trading Model")

page = st.sidebar.selectbox(
    "Choose page",
    ["Single Stock Analysis", "Watchlist Ranking", "Chart Viewer"],
)

use_cache = st.sidebar.checkbox("Use cached yfinance data", value=True)

if page == "Single Stock Analysis":
    ticker = st.text_input("Enter stock ticker", "AAPL").upper()

    if st.button("Analyze"):
        try:
            data = get_price_data(
                ticker, period="6mo", interval="1d", use_cache=use_cache
            )
            data = add_indicators(data)

            latest = data.dropna().iloc[-1]

            st.subheader(f"{ticker} Summary")
            st.write(f"Latest close: ${latest['close']:.2f}")
            st.write(f"RSI 14: {latest['RSI_14']:.2f}")
            st.write(f"5-day return: {latest['return_5d'] * 100:.2f}%")
            st.write(f"20-day return: {latest['return_20d'] * 100:.2f}%")
            st.write(f"Volume ratio: {latest['volume_ratio']:.2f}")

            st.subheader("Recent Data")
            st.dataframe(data.tail())

            st.subheader("Close Price")
            st.line_chart(data["close"])

            st.subheader("Moving Averages")
            st.line_chart(data[["close", "SMA_20", "SMA_50", "EMA_20"]])

            st.subheader("RSI")
            st.line_chart(data["RSI_14"])

            st.subheader("Volume")
            st.bar_chart(data["volume"])

        except Exception as error:
            st.error(f"Could not analyze {ticker}: {error}")


elif page == "Watchlist Ranking":
    tickers_text = st.text_area(
        "Enter tickers separated by commas",
        "AAPL, MSFT, NVDA, AMD, META, GOOGL, AMZN, TSLA",
    )

    tickers = [
        ticker.strip().upper() for ticker in tickers_text.split(",") if ticker.strip()
    ]

    if st.button("Rank Watchlist"):
        try:
            with st.spinner("Downloading/loading price data..."):
                price_data = get_multiple_price_data(
                    tickers,
                    period="6mo",
                    interval="1d",
                    use_cache=use_cache,
                )

            ranking = rank_stocks(price_data)

            st.subheader("Ranking Results")
            st.dataframe(ranking, use_container_width=True, hide_index=True)

            st.subheader("Top Candidates")
            st.dataframe(ranking.head(5), use_container_width=True, hide_index=True)

        except Exception as error:
            st.error(f"Could not rank watchlist: {error}")

elif page == "Chart Viewer":
    st.subheader("Chart Viewer")

    cached_tickers = get_cached_tickers()

    chart_source = st.radio(
        "Choose chart source",
        ["Enter ticker manually", "Choose from loaded/cache tickers"],
    )

    if chart_source == "Choose from loaded/cache tickers" and cached_tickers:
        ticker = st.selectbox("Loaded tickers", cached_tickers)
    else:
        ticker = st.text_input("Enter stock ticker", "AAPL").upper()

    period = st.selectbox(
        "Chart period",
        ["3mo", "6mo", "1y", "2y"],
        index=1,
    )

    support_lookback = st.slider(
        "Support/resistance lookback days",
        min_value=20,
        max_value=120,
        value=60,
        step=10,
    )

    st.sidebar.subheader("Chart overlays")

    show_ema_10 = st.sidebar.checkbox("EMA 10", value=True)
    show_ema_20 = st.sidebar.checkbox("EMA 20", value=True)
    show_ema_50 = st.sidebar.checkbox("EMA 50", value=True)
    show_ema_100 = st.sidebar.checkbox("EMA 100", value=False)
    show_ema_200 = st.sidebar.checkbox("EMA 200", value=True)
    show_sma_200 = st.sidebar.checkbox("SMA 200", value=True)
    show_support_resistance = st.sidebar.checkbox("Support / Resistance", value=True)

    if st.button("Load Chart"):
        try:
            data = get_price_data(
                ticker,
                period=period,
                interval="1d",
                use_cache=use_cache,
            )

            chart = create_trading_chart(
                data,
                ticker=ticker,
                support_lookback=support_lookback,
                show_ema_10=show_ema_10,
                show_ema_20=show_ema_20,
                show_ema_50=show_ema_50,
                show_ema_100=show_ema_100,
                show_ema_200=show_ema_200,
                show_sma_200=show_sma_200,
                show_support_resistance=show_support_resistance,
            )

            st.plotly_chart(chart, use_container_width=True)

            latest = data.iloc[-1]

            st.subheader(f"{ticker} Latest Data")
            st.write(f"Close: ${latest['close']:.2f}")
            st.write(f"High: ${latest['high']:.2f}")
            st.write(f"Low: ${latest['low']:.2f}")
            st.write(f"Volume: {latest['volume']:,.0f}")

        except Exception as error:
            st.error(f"Could not load chart for {ticker}: {error}")

    if cached_tickers:
        st.caption(f"Loaded/cache tickers available: {', '.join(cached_tickers)}")
