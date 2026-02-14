import yfinance as yf
import pandas as pd

def fetch_data(ticker, start_date, end_date):
    """Downloads data from YF and fixes missing values."""
    print(f"Downloading {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)['Close']
    data.ffill(inplace=True)
    return data