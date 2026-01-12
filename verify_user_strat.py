import pandas as pd
import strategy
import backtest
import data_loader
import numpy as np

# 1. Load Data
df = data_loader.load_data()
if df is None:
    print("Error: Could not load data.")
    exit()

# 2. Get Indicators and Signals
ind = strategy.get_expanded_indicators(df)
repo = strategy.SignalRepository(ind)

# Get sorted keys as used in mining
keys, sig_mat, _ = repo.get_signal_matrix()

# 3. Define Strategy Target
# EMA_9_Cross_20_Down (Index 22) AND Ichimoku_TK_Cross_Bull (Index 32)
# Let's verify indexes
s1_name = "EMA_9_Cross_20_Down"
s2_name = "Ichimoku_TK_Cross_Bull"

print(f"Index for {s1_name}: {keys.index(s1_name)}")
print(f"Index for {s2_name}: {keys.index(s2_name)}")

# Symmetric signals for Short
sym_s1_name, sym_s2_name, _ = repo.get_symmetric_logic(s1_name, s2_name, "AND")
print(f"Short Entry Signals: {sym_s1_name} AND {sym_s2_name}")

# 4. Construct Signals
long_signal = repo.evaluate_combined_signal(s1_name, s2_name, "AND")
short_signal = repo.evaluate_combined_signal(sym_s1_name, sym_s2_name, "AND")

# No neutral filter (Index -1)
# No specific exit signal other than SL/TP and Reverse

# 5. Run VectorBT Portfolio Simulation (Replicating Mining Logic)
# Use the same parameters as in Hall of Fame:
# SL: 0.03876637370747484, TP: 0.1, Leverage: 1
leverage = 1
sl = 0.03876637370747484
tp = 0.1

import vectorbt as vbt
# Replicating the logic in generate_and_test precisely
# Note: Mining evaluates on Train and Test separately. 
# Here we will run on the ENTIRE dataset to see current performance.

print("\n--- Running Full Dataset Backtest ---")
pf = vbt.Portfolio.from_signals(
    df['close'], 
    entries=long_signal, 
    short_entries=short_signal,
    exits=None, # Exit handled by SL/TP and Reverse
    short_exits=None,
    freq='5m', 
    init_cash=10000, 
    fees=0.0004, 
    sl_stop=sl, 
    tp_stop=tp,
    upon_opposite_entry='reverse'
)

print(f"Total Return: {pf.total_return() * leverage * 100:.2f}%")
print(f"Max Drawdown: {pf.max_drawdown() * leverage * 100:.2f}%")
print(f"Win Rate: {pf.trades.win_rate() * 100:.2f}%")
print(f"Trade Count: {pf.trades.count()}")
print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")

# 6. Check Monthly Return
test_days = (df.index[-1] - df.index[0]).total_seconds() / (24 * 3600)
monthly_ret = ((1 + (pf.total_return() * leverage)) ** (30 / test_days)) - 1
print(f"Monthly Return: {monthly_ret * 100:.2f}%")
