import os
import yfinance as yf
import pandas as pd

# Define the folder name relative to the project root
DATA_DIR = "data"

def fetch_data(tickers, start_date, end_date, filename="market_data.csv"):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created directory: {DATA_DIR}")

    file_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(file_path):
        print(f"✅ Found local data at '{file_path}'. Loading...")
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
    else:
        print(f"⚠️ File not found. Downloading {tickers} from Yahoo Finance...")
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Close']
        data.ffill(inplace=True)
        
        data.to_csv(file_path)
        print(f"💾 Saved data to '{file_path}'")

    return data