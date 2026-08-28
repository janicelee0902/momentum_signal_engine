import matplotlib.pyplot as plt

from data import fetch_price_data
from signals import compute_momentum, select_top_stocks
from backtest import simulate_portfolio, compute_sharpe_ratio, compute_max_drawdown, evaluate_period

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

main()