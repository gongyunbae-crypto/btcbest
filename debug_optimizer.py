import pandas as pd
import numpy as np
import data_loader
import strategy
import backtest
import vectorbt as vbt

def run_debug_optimization():
    print("--- Loading Data ---")
    df = data_loader.load_data()
    if df is None:
        print("Data not found. Fetching...")
        df = data_loader.fetch_binance_futures_data("BTC/USDT", timeframe="1h", since_years=1)
    
    print(f"Data loaded: {len(df)} records")
    
    # Run Strategy
    print("--- Generating Signals ---")
    daily_levels = strategy.get_daily_levels(df)
    long_signals, short_signals, long_exits, short_exits = strategy.apply_strategy_signals(df, daily_levels)
    
    # Run Optimizer
    print("--- Running Optimization (Trailing Stop = 0.5%) ---")
    # Simulate user selecting 0.5% trailing stop
    trailing_stop = 0.005 
    
    try:
        results = backtest.optimize_parameters(
            df, 
            long_signals, 
            short_signals, 
            long_exits, 
            short_exits, 
            trailing_stop=trailing_stop
        )
        
        print("\n--- Optimization Results ---")
        print(results)
        
        if results['Total Return'] <= -100:
            print("\n❌ FAILED: Best Result is still LIQUIDATION.")
        elif results['Total Return'] > 0:
            print("\n✅ SUCCESS: Found profitable parameters!")
        else:
            print("\n⚠️ WARNING: Best result is negative but not liquidated.")
            
    except Exception as e:
        print(f"\n❌ ERROR during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug_optimization()
