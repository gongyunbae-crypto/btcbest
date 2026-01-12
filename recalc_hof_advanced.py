import pandas as pd
import data_loader
import backtest
import os

print("Applying Advanced Metrics to Hall of Fame...")

# Load Data
df = data_loader.load_data("btc_futures_data_5m.csv")
if df is None:
    print("Fetching data...")
    df = data_loader.fetch_binance_futures_data("BTC/USDT", timeframe="5m", since_years=1)
    data_loader.save_data(df, "btc_futures_data_5m.csv")

if df is not None:
    composer = backtest.RandomStrategyComposer(df)
    
    # Recalculate
    composer.recalculate_hof_metrics()
    print("✅ Hall of Fame updated with Sharpe, Sortino, Profit Factor, Max Cons Loss!")
else:
    print("❌ Failed to load data.")
