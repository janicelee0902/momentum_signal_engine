import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

data = yf.download(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "XOM", "JNJ", "PG", "NVDA"], start="2020-01-01", end="2024-01-01")
momentum = data["Close"].pct_change(periods=126)
monthly_momentum = momentum.resample("ME").last()

# Rank 1 = highest momentum. Using ascending=False since pandas' default ranks smallest values as 1.
ranks = monthly_momentum.rank(axis=1, ascending=False)
#print(ranks.head(10))

# Deciding which stocks to hold
selected = ranks <= 3
#print(selected.head(10))

# Eliminating lookahead bias
tradeable_selected = selected.shift(1).fillna(False)
#print(tradeable_selected.head(10))

# Portfolio simulation
monthly_prices = data["Close"].resample("ME").last()
monthly_returns = monthly_prices.pct_change()
#print(monthly_returns.head(10))

# averaging the returns of stocks that we are holding (gross return - doesn't factor in transaction cost)
portfolio_returns = monthly_returns.where(tradeable_selected).mean(axis=1)
#print(portfolio_returns.head(10))

# cumalative returns
cumulative_returns = (1 + portfolio_returns).cumprod()
#print(cumulative_returns.head(15))

# to detect stocks that were just bought or sold
previous_holdings = tradeable_selected.shift(1).fillna(False)
bought = tradeable_selected & ~previous_holdings # previous holdings - False & current holdings - True
sold = ~tradeable_selected & previous_holdings # previous holdings - True & current holdings - False

# deduct total transaction cost 
num_trades = bought.sum(axis=1) + sold.sum(axis=1)
cost_rate = 0.001
position_size_fraction = 1/3
transaction_costs = num_trades * cost_rate * position_size_fraction

# net return and cumalative returns
net_portfoilio_returns = portfolio_returns - transaction_costs
net_cumulative_returns = (1 + net_portfoilio_returns).cumprod()
print(cumulative_returns[20:25])
print(net_cumulative_returns[20:25])

plt.plot(cumulative_returns, label="Gross returns")
plt.plot(net_cumulative_returns, label="Net returns (after costs)")
plt.legend()
plt.show()