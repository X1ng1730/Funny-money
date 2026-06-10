import streamlit as st

from alpha_vantage import get_daily_prices, get_rsi

st.title("Swing Trading Model")

ticker = st.text_input("Enter stock ticker", "AAPL").upper()


@st.cache_data(ttl=900)
def load_alpha_vantage_data(symbol: str):
    prices = get_daily_prices(symbol, outputsize="compact")
    rsi = get_rsi(symbol)
    return prices, rsi


if st.button("Analyze"):
    try:
        data, rsi_data = load_alpha_vantage_data(ticker)
    except Exception as exc:
        st.error(f"Could not load Alpha Vantage data for {ticker}: {exc}")
        st.stop()

    latest_close = float(data["close"].iloc[-1])
    latest_rsi = float(rsi_data["RSI"].iloc[-1])

    st.subheader(f"{ticker} recent Alpha Vantage daily data")
    st.dataframe(data.tail())

    st.subheader("Simple summary")
    st.write(f"Latest close price: ${latest_close:.2f}")
    st.write(f"Latest RSI (14): {latest_rsi:.2f}")

    st.subheader("Price chart")
    st.line_chart(data["close"])

    st.subheader("RSI chart")
    st.line_chart(rsi_data["RSI"])
