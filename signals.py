import pandas as pd

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