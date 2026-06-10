import yfinance as yf

ticker = "AAPL"

data = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)

print(data.tail())
