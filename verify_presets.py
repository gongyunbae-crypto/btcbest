
import pandas as pd
import vectorbt as vbt
import strategy
import data_loader

# Mock Data
df = data_loader.load_data("btc_futures_data_5m.csv")
if df is None:
    print("No data found, fetching...")
    df = data_loader.fetch_binance_futures_data("BTC/USDT", timeframe="5m", since_years=1)

print("Data Loaded:", df.shape)

# Test Golden Cross
print("\n--- Testing Golden Cross ---")
# Need to implement logic first? No, I will implement it in strategy.py then run this.
# This file is just a placeholder for now to verify I can import strategy.
print("Strategy module imported.")
