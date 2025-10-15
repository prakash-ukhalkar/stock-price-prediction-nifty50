# data/download_data.py

import pandas as pd
from nsepy import get_history
from datetime import date
import os
import yfinance as yf

def fetch_nifty50_data(start_date=date(2018, 1, 1), end_date=date.today()):
    """
    Downloads NIFTY 50 historical data using nsepy or falls back to yfinance.
    
    Args:
        start_date (date): The start date for data download.
        end_date (date): The end date for data download.
    
    Returns:
        pd.DataFrame: DataFrame containing NIFTY 50 OHLCV data.
    """
    
    # --- Try NSEPY first (Primary choice for NSE data) ---
    try:
        print(f"Attempting to download NIFTY 50 data from NSEPY ({start_date} to {end_date})...")
        df = get_history(symbol='NIFTY',
                        start=start_date,
                        end=end_date,
                        index=True)
        if df.empty:
            raise ValueError("NSEPY returned an empty DataFrame. Trying yfinance fallback.")
        print("NSEPY download successful.")
        
    except Exception as e:
        print(f"NSEPY failed or returned empty: {e}. Falling back to YFinance...")
        
        # --- Fallback to YFinance ---
        ticker = '^NSEI'  # Yahoo Finance ticker for NIFTY 50
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            print("YFinance also returned an empty DataFrame. Please check the dates/connection.")
            return pd.DataFrame()
        
    # Standardize columns (YFinance uses 'Adj Close', NSEPY uses 'Close')
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    
    # Clean up and ensure Date is the index
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
    df.index.name = 'Date'
    
    return df

def save_nifty50_data(df, file_path='data/processed/nifty50_clean.csv'):
    """Saves the DataFrame to a specified CSV path."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path)
    print(f"Data successfully saved to {file_path}")

if __name__ == '__main__':
    # Default behavior: Download last 6 years of data
    START = date(date.today().year - 6, 1, 1)
    END = date.today()
    
    df_new = fetch_nifty50_data(start_date=START, end_date=END)
    
    if not df_new.empty:
        save_nifty50_data(df_new)