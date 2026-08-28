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
    """Download historical daily price data for the given tickers and date range."""
    return yf.download(tickers, start=start, end=end)

def compute_momentum(data, lookback_days):
    """
    Compute the cross-sectional momentum signal for each stock, resampled to monthly.

    Momentum is the percentage return over the trailing 'looback_days' trading days.
    Resampled to the last trading day of each month to match the montly rebalancing schedule.

    Parameters:
        data: DataFrame of price data with a 'Close' column per ticker
        lookback_days: number of trading days to look back

    Returns:
        DataFrame of monthly momentum values, one column per ticker
    """
    momentum = data["Close"].pct_change(periods=lookback_days)
    monthly_momentum = momentum.resample("ME").last()
    return monthly_momentum
    
def select_top_stocks(monthly_momentum, top_n):
    """
    Select the top stocks to hold for each month.
    Rank 1 = highest momentum. Using ascending=False since pandas' default ranks smallest values as 1.
    The returned holdings for any given month were decided using the previous month's data to eliminate lookahead bias.

    Parameters:
        monthly_momentum: the momentum signal for each stock every month
        top_n: the number of top stocks to hold

    Returns:
        A boolean DataFrame indicating which stocks are held each month
    """
    ranks = monthly_momentum.rank(axis=1, ascending=False)
    selected = ranks <= top_n
    tradeable_selected = selected.shift(1).fillna(False).astype(bool) # Eliminating lookahead bias
    return tradeable_selected

def simulate_portfolio(data, tradeable_selected, cost_rate, top_n):
    """Run the full portfolio simulation: gross/net returns and their cumulative series."""
    monthly_prices = data["Close"].resample("ME").last()
    portfolio_returns = compute_portfolio_returns(monthly_prices, tradeable_selected)
    transaction_costs = compute_transaction_costs(tradeable_selected, cost_rate, top_n)

    cumulative_returns = (1 + portfolio_returns).cumprod()
    net_portfolio_returns = portfolio_returns - transaction_costs
    net_cumulative_returns = (1 + net_portfolio_returns).cumprod()

    return portfolio_returns, net_portfolio_returns, cumulative_returns, net_cumulative_returns
     
def compute_portfolio_returns(monthly_prices, tradeable_selected):
    """Compute equal-weighted monthly portfolio returns from held stocks' prices."""
    monthly_returns = monthly_prices.pct_change()
    return monthly_returns.where(tradeable_selected).mean(axis=1)

def compute_transaction_costs(tradeable_selected, cost_rate, top_n):
    """
    Estimate monthly transaction costs from portfolio turnover.

    Assumes a flat cost_rate (e.g. 0.001 = 0.1%) per trade, applied to a single
    position's share of the portfolio (1/top_n), since only the traded stocks'
    share of the portfolio incurs a cost, not the whole portfolio. This is a
    simplification — real costs vary by broker, order size, and liquidity.

    Parameters:
        tradeable_selected: boolean DataFrame of monthly holdings
        cost_rate: assumed cost per trade, as a fraction (e.g. 0.001)
        top_n: number of stocks held, used to size each position

    Returns:
        Series of estimated transaction cost (as a fraction of portfolio) per month
    """
    bought, sold = detect_trades(tradeable_selected)
    num_trades = bought.sum(axis=1) + sold.sum(axis=1)
    position_size_fraction = 1/top_n
    return num_trades * cost_rate * position_size_fraction

def detect_trades(tradeable_selected):
    """
    Identify which stocks were bought or sold each month, by comparing current
    holdings against the previous month's holdings.

    Parameters:
        tradeable_selected: boolean DataFrame of monthly holdings

    Returns:
        (bought, sold): two boolean DataFrames flagging newly bought/sold stocks
    """
    previous_holdings= tradeable_selected.shift(1).fillna(False).astype(bool)
    bought = tradeable_selected & ~previous_holdings
    sold = ~tradeable_selected & previous_holdings
    return bought, sold

def evaluate_period(returns, start_date, end_date):
    """Compute cumulative compounded returns for a specific date range only."""
    return (1 + returns[start_date : end_date]).cumprod()

def compute_sharpe_ratio(returns, risk_free_rate=0):
    """
    Compute the annualized Sharpe ratio (risk-adjusted return) for a return series.

    Monthly mean return is annualized by x12; monthly standard deviation is
    annualized by x sqrt(12), since variance (not standard deviation) scales
    linearly with time. Assumes a risk-free rate of 0 as a simplification.

    Parameters:
        returns: Series of periodic (monthly) returns
        risk_free_rate: assumed risk-free rate, default 0

    Returns:
        Annualized Sharpe ratio as a float
    """
    annualised_return = returns.mean() * 12
    annualised_std = returns.std() * (12 ** 0.5)
    return (annualised_return - risk_free_rate) / annualised_std

def compute_max_drawdown(cumulative_returns):
    """
    Compute the maximum peak-to-trough decline in cumulative returns.

    Tracks the running historical peak at each point, then finds the largest
    percentage drop from any peak to a subsequent low.

    Parameters:
        cumulative_returns: Series of compounded cumulative returns

    Returns:
        Maximum drawdown as a negative float (e.g. -0.30 = a 30% decline)
    """
    running_peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_peak) / running_peak
    return drawdown.min()

main()