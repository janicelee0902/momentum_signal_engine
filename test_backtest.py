import pandas as pd
from explore import detect_trades, compute_max_drawdown, select_top_stocks

def test_detect_trades():
    sample_holdings = pd.DataFrame({
        "AAPL": [True, True, False],
        "MSFT": [False, True, True]
    }, dtype=bool)
    expected_bought = pd.DataFrame({
        "AAPL": [True, False, False],
        "MSFT": [False, True, False]
    })
    expected_sold = pd.DataFrame({
        "AAPL": [False, False, True],
        "MSFT": [False, False, False]
    })
    bought, sold = detect_trades(sample_holdings)

    pd.testing.assert_frame_equal(bought, expected_bought)
    pd.testing.assert_frame_equal(sold, expected_sold)

def test_compute_max_drawdown():
    sample = pd.Series([1.0, 1.5, 1.2, 1.8, 0.9])
    expected_max_drawdown = -0.5
    max_drawdown = compute_max_drawdown(sample)
    assert max_drawdown == expected_max_drawdown

def test_select_top_stocks():
    sample_momentum = pd.DataFrame({
        "AAPL": [0.10, 0.30],
        "MSFT": [0.20, 0.10],
        "GOOGL": [0.05, 0.25]
    })
    expected_top_stocks = pd.DataFrame({
        "AAPL": [False, True],
        "MSFT": [False, True],
        "GOOGL": [False, False]
    }, dtype=bool)
    top_stocks = select_top_stocks(sample_momentum, 2)
    pd.testing.assert_frame_equal(top_stocks, expected_top_stocks)
