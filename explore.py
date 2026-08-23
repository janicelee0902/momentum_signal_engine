import pandas as pd
import yfinance as yf

data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
momentum = data["Close"]["AAPL"].pct_change(periods=126)
print(momentum[125:130])

#print(pd.Series([10, 11, 12, 10, 13]).pct_change(periods = 2))
