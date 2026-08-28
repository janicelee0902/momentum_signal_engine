import yfinance as yf

def fetch_price_data(tickers, start, end):
    """Download historical daily price data for the given tickers and date range."""
    return yf.download(tickers, start=start, end=end)