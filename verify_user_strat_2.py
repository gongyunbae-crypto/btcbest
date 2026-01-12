import pandas as pd
import strategy
import data_loader
import numpy as np
import vectorbt as vbt

# 1. Load Data
df = data_loader.load_data()
if df is None:
    print("Error: Could not load data.")
    exit()

# 2. Get Indicators and Signals
ind = strategy.get_expanded_indicators(df)
repo = strategy.SignalRepository(ind)

# Define Strategy Target (Vortex_Bullish_Cross AND KST_Bullish_Cross)
s1_name = "Vortex_Bullish_Cross"
s2_name = "KST_Bullish_Cross"

# Get Symmetric Signals for Short
sym_s1_name, sym_s2_name, _ = repo.get_symmetric_logic(s1_name, s2_name, "AND")

# 3. Construct Signals
long_signal = repo.evaluate_combined_signal(s1_name, s2_name, "AND")
short_signal = repo.evaluate_combined_signal(sym_s1_name, sym_s2_name, "AND")

# 4. Define Parameters from saved_strategies.json
leverage = 2
sl = 0.02850182943782375
tp = 0.1

print(f"--- Running Full Dataset Backtest for: {s1_name} AND {s2_name} ---")
print(f"Params: Leverage={leverage}, SL={sl*100:.2f}%, TP={tp*100:.2f}%")

pf = vbt.Portfolio.from_signals(
    df['close'], 
    entries=long_signal, 
    short_entries=short_signal,
    exits=None, 
    short_exits=None,
    freq='5m', 
    init_cash=10000, 
    fees=0.0004, 
    sl_stop=sl, 
    tp_stop=tp,
    upon_opposite_entry='reverse'
)

total_ret = pf.total_return() * leverage 
max_drawdown = pf.max_drawdown() * leverage

print(f"Total Return: {total_ret * 100:.2f}%")
print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
print(f"Win Rate: {pf.trades.win_rate() * 100:.2f}%")
print(f"Trade Count: {pf.trades.count()}")
print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")

# Check Monthly Return
test_days = (df.index[-1] - df.index[0]).total_seconds() / (24 * 3600)
monthly_ret = ((1 + total_ret) ** (30 / test_days)) - 1
print(f"Monthly Return: {monthly_ret * 100:.2f}%")
