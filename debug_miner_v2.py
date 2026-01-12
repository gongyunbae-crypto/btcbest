import pandas as pd
import numpy as np
import backtest
import data_loader

def test_mining_engine():
    print("--- 1. Loading Data ---")
    df = data_loader.load_data()
    if df is None:
        print("Data not found locally. Fetching small sample...")
        df = data_loader.fetch_binance_futures_data("BTC/USDT", since_years=1)
    
    print(f"Data Loaded: {len(df)} records")

    print("--- 2. Initializing Engine ---")
    try:
        engine = backtest.StrategyMiningEngine(df)
        print("Engine Initialized.")
    except Exception as e:
        print(f"FAILED to initialize engine: {e}")
        return

    print("--- 3. Running Mining Simulation ---")
    # Use a small grid for testing
    sweeps = np.array([0.005, 0.01]) # 0.5%, 1.0%
    rsis = np.array([30, 40])
    levs = np.array([1, 5])
    
    try:
        results = engine.run_mining(
            sweep_depths=sweeps,
            rsi_thresholds=rsis,
            leverages=levs
        )
        
        print("\n--- 4. Mining Results ---")
        if results.empty:
            print("Mining finished but returned NO results (all filtered?).")
        else:
            print(f"Success! Found {len(results)} strategies.")
            print("Top 3 Strategies:")
            print(results.head(3)[['Description', 'Return', 'Sharpe', 'MDD']])
            
            # Check for sanity
            best = results.iloc[0]
            if best['MDD'] < -100:
                print("❌ CRITICAL: Liquidated strategy found in results!")
            else:
                print("✅ Sanity Check Passed: Top strategy is not liquidated.")
                
    except Exception as e:
        print(f"❌ MINING FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mining_engine()
