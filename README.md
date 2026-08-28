# Momentum Signal Engine

A momentum-based trading signal and backtesting engine, built as a learning project


## Motivation

I wanted to build this project to understand how a quantitative trading strategy is constructed and evaluated. With no prior finance background, I treated this project as a genuine, deliberate learning exercise to understand finance concepts such as returns, momentum, rebalancing, transaction costs, and the Sharpe ratio, as well as to refine my Python skills and familiarity with the pandas library. I made sure to avoid lookahead bias and account for transaction costs, producing an honest evaluation beyond raw returns, so that this project reflects the same rigor as a real backtest as much as possible.


## Methodology

A long-only cross-sectional momentum strategy was applied to 10 large-cap US stocks across different sectors, including AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, XOM, JNJ, PG, and NVDA. Daily closing prices from 2020-01-01 to 2024-01-01 were used, sourced via yfinance. The momentum signal was computed as the percentage return over a 126-trading-day (~6 month) lookback window, and the portfolio is rebalanced monthly using the prior month's signal (to avoid lookahead bias). Each month, the top 3 of 10 stocks by momentum rank are held, equally weighted across positions. A flat 0.1% transaction cost was applied per trade, for simplicity. Performance was evaluated using gross and net cumulative returns, sub-period consistency checks, the Sharpe ratio, and maximum drawdown.


## Design Decisions

**Ranking convention**: Rank 1 represents the highest momentum stock, using `ascending=False` since pandas' default ranking treats the smallest value as rank 1 — the opposite of what's needed here.

**Lookahead bias**: There is a risk of including values in a calculation that have not actually occurred yet — e.g. deciding July's holdings using July's own incomplete data. This is solved by using `.shift(1)`, so each month's holdings are decided using the prior month's complete data.

**Equal-weighting assumption**: A simplification was made, treating the top three stocks equally despite their different momentum signal strengths. A real fund might weight holdings by each stock's respective signal strength instead.

**Transaction cost assumption**: A flat cost of 0.1% of position value per trade (buy or sell) was applied. This is a simplified but standard assumption for backtesting liquid large-cap stocks. A flat rate lets the backtest reflect the cost of rebalancing without introducing too many complications, as real-world costs vary by broker, order size, and stock liquidity. Trades were detected by comparing each month's holdings against the previous month's holdings. Costs are scaled by position size (1/3 per stock) rather than applied to the whole portfolio per trade.

**Handling missing values in shifted boolean data**: At the very start of the simulation, there are no prior holdings to compare against, so `.fillna(False)` was used to treat "no prior holding data" as "not held" — a factually correct interpretation, not just a technical workaround. However, `.shift()` on a boolean DataFrame can silently change its underlying data type (likely from introducing `NaN` at the shift boundary), which caused `~` (logical NOT) to behave incorrectly rather than as a clean negation. This was fixed by explicitly forcing the type back with `.astype(bool)` after `.fillna(False)`. The root cause in pandas' internals was not fully traced, but the fix was confirmed correct via unit testing.

**No fitted parameters**: The strategy is fixed and rule-based, so classic ML-style walk-forward validation (retrain then test on new data) doesn't directly apply. Instead, the robustness of this strategy was tested across distinct sub-periods within the original four-year window, to check the effect isn't just due to one unusual stretch, and whether the strategy makes money consistently.

**Named constants**: `LOOKBACK_DAYS = 126` and `TOP_N = 3` are defined explicitly rather than hardcoded, so choices are visible and easy to change.

**Unit testing**: Hand-verified expected outputs were written for the most critical functions (`detect_trades`, `compute_max_drawdown`, `select_top_stocks`), which caught and fixed two bugs during development. 


## Results

Over the full four-year backtest period (2020-01-01 to 2024-01-01), the strategy achieved a gross cumulative return of ~2.8x (before transaction costs). The net cumulative return was comparable, however the transaction costs produced a modest but consistent drag (~1.3% by ~17 months in), rather than a dramatic one, consistent with relatively low monthly turnover.

Performance was not uniform across sub-periods. The strategy gained substantially in 2020-2021 (~+119%, ending at 2.19x), lost value in 2022 (~-13%, ending at 0.87x), and recovered in 2023-2024 (~+45%, ending at 1.45x). This pattern is consistent with momentum's known vulnerability to sharp market reversals, and highlights that the strong aggregate 4-year return was concentrated in specific periods rather than earned uniformly across the three sub-periods.

A Sharpe ratio of ~1.02 indicated decent but not exceptional risk-adjusted returns, consistent with the strategy's high volatility observed in the sub-period analysis (strong gains in 2020-21, losses in 2022).

Maximum drawdown of ~30% indicates this strategy experienced a substantial decline at some point, likely coinciding with 2022's poor performance.

The equity curve (gross vs net) is shown below:
![alt text](equity_curve.png)


## Limitations

- **Small, hand-picked universe**: built around 10 large-cap US stocks, which may not be representative of a broader market or a different stock selection.
- **Unusual market conditions**: the backtest period includes the COVID crash/recovery and the 2022 downturn, so performance may not reflect "normal" market conditions.
- **No fitted parameters**: classic walk-forward validation doesn't directly apply, since there's nothing to retrain. Sub-period testing was used as a substitute, which is reasonable but less rigorous.
- **Simplified transaction cost model**: doesn't account for slippage, market impact, or varying liquidity across stocks.
- **Free data source**: yfinance is not institutional-grade data, so there may be potential data quality or adjustment issues.
- **Equal-weighting simplification**: doesn't account for differing conviction or signal strength across holdings.


## How to Run It

1. Clone the repository:
   `git clone https://github.com/janicelee0902/momentum_signal_engine.git`
2. Create and activate a virtual environment:
   `python -m venv venv`
   `source venv/Scripts/activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run the backtest:
   `python main.py`
5. (Optional) Run the test suite:
   `pytest`