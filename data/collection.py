import yfinance as yf

data = yf.download("AAPL",
                   start="2015-01-01",
                   end="2025-01-01")

print(data.head())
data.to_csv("aapl.csv")