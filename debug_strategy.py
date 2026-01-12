import pandas as pd
import numpy as np
import vectorbt as vbt
import data_loader
import strategy
import backtest

# Load small chunk of data (or generate dummy)
# Let's try to load the actual data file if it exists
try:
    df = data_loader.load_data()
    if df is None:
        print("No data found, fetching small amount...")
        df = data_loader.fetch_binance_futures_data("BTC/USDT", timeframe="1h", since_years=1)
except Exception as e:
    print(f"Error loading data: {e}")
    # Dummy
    df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=1000, freq='1h'))
    df['close'] = np.random.randn(1000).cumsum() + 100
    df['open'] = df['close']
    df['high'] = df['close'] + 1
    df['low'] = df['close'] - 1
    df['volume'] = 100

# Run Strategy
daily_levels = strategy.get_daily_levels(df)
long_signals, short_signals, long_exits, short_exits = strategy.apply_strategy_signals(df, daily_levels)

print(f"Data Length: {len(df)}")
print(f"Long Signals: {long_signals.sum()}")
print(f"Short Signals: {short_signals.sum()}")
print(f"Long Exits: {long_exits.sum()}")
print(f"Short Exits: {short_exits.sum()}")

# Check alignment
print("\n--- Sample Signals ---")
print(long_signals[long_signals].head())

# Run Backtest (1x, No Fees, No SL) to see raw interactions
pf = vbt.Portfolio.from_signals(
    df['close'],
    entries=long_signals,
    short_entries=short_signals,
    exits=long_exits,
    short_exits=short_exits,
    freq='1h',
    fees=0.0004
)

print("\n--- Backtest Stats (raw) ---")
print(f"Total Trades: {pf.trades.count()}")
print(f"Win Rate: {pf.trades.win_rate()}")
print(f"Total Return: {pf.total_return()}")
print(f"Max Drawdown: {pf.max_drawdown()}")
