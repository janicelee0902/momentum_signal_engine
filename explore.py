import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

def main():
    LOOKBACK_DAYS = 126
    TOP_N = 3
    COST_RATE = 0.001

    data = fetch_price_data(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "XOM", "JNJ", "PG", "NVDA"], start="2020-01-01", end="2024-01-01")
    monthly_momentum = compute_momentum(data, LOOKBACK_DAYS)
    tradeable_selected = select_top_stocks(monthly_momentum, TOP_N)
    portfolio_returns, net_portfolio_returns, cumulative_returns, net_cumulative_returns = simulate_portfolio(data, tradeable_selected, COST_RATE, TOP_N)

    plt.plot(cumulative_returns, label="Gross returns")
    plt.plot(net_cumulative_returns, label="Net returns (after costs)")
    plt.legend()
    plt.show()

    period1 = evaluate_period(net_portfolio_returns, "2020-01-01", "2021-12-31")
    period2 = evaluate_period(net_portfolio_returns, "2022-01-01", "2022-12-31")
    period3 = evaluate_period(net_portfolio_returns, "2023-01-01", "2024-01-01")

    print(f"2020-2021: {period1.iloc[-1]}")
    print(f"2022: {period2.iloc[-1]}")
    print(f"2023-2024: {period3.iloc[-1]}")

    sharpe = compute_sharpe_ratio(net_portfolio_returns)
    print(f"Sharpe ratio: {sharpe}")

    max_drawdown = compute_max_drawdown(net_cumulative_returns)
    print(f"Maximum Drawdown: {max_drawdown}")

def fetch_price_data(tickers, start, end):
    return yf.download(tickers, start=start, end=end)

def compute_momentum(data, lookback_days):
    # returning the monthly resampled momentum
    momentum = data["Close"].pct_change(periods=lookback_days)
    monthly_momentum = momentum.resample("ME").last()
    return monthly_momentum
    
def select_top_stocks(monthly_momentum, top_n):
    # Rank 1 = highest momentum. Using ascending=False since pandas' default ranks smallest values as 1.
    ranks = monthly_momentum.rank(axis=1, ascending=False)
    selected = ranks <= top_n
    tradeable_selected = selected.shift(1).fillna(False) # Eliminating lookahead bias
    return tradeable_selected

def simulate_portfolio(data, tradeable_selected, cost_rate, top_n):
    monthly_prices = data["Close"].resample("ME").last()
    portfolio_returns = compute_portfolio_returns(monthly_prices, tradeable_selected)
    transaction_costs = compute_transaction_costs(tradeable_selected, cost_rate, top_n)

    cumulative_returns = (1 + portfolio_returns).cumprod()
    net_portfolio_returns = portfolio_returns - transaction_costs
    net_cumulative_returns = (1 + net_portfolio_returns).cumprod()

    return portfolio_returns, net_portfolio_returns, cumulative_returns, net_cumulative_returns
     
def compute_portfolio_returns(monthly_prices, tradeable_selected):
    monthly_returns = monthly_prices.pct_change()
    return monthly_returns.where(tradeable_selected).mean(axis=1)

def compute_transaction_costs(tradeable_selected, cost_rate, top_n):
    bought, sold = detect_trades(tradeable_selected)
    num_trades = bought.sum(axis=1) + sold.sum(axis=1)
    position_size_fraction = 1/top_n
    return num_trades * cost_rate * position_size_fraction

def detect_trades(tradeable_selected):
    previous_holdings= tradeable_selected.shift(1).fillna(False)
    bought = tradeable_selected & ~previous_holdings
    sold = ~tradeable_selected & previous_holdings
    return bought, sold

def evaluate_period(returns, start_date, end_date):
    # computes cumulative returns of specific period
    return (1 + returns[start_date : end_date]).cumprod()

def compute_sharpe_ratio(returns, risk_free_rate=0):
    annualised_return = returns.mean() * 12
    annualised_std = returns.std() * (12 ** 0.5)
    return (annualised_return - risk_free_rate) / annualised_std

def compute_max_drawdown(cumulative_returns):
    running_peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_peak) / running_peak
    return drawdown.min()

main()