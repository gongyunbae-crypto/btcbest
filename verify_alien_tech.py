
import pandas as pd
import backtest
import data_loader

def test_alien_tech():
    print("--- 1. Loading Data (5m) ---")
    df = data_loader.load_data("btc_futures_data_5m.csv")
    if df is None:
        print("Data not found, fetching...")
        df = data_loader.fetch_binance_futures_data(timeframe="5m", since_years=1)
    
    print(f"Data Loaded: {len(df)} rows")
    
    print("--- 2. Initializing Composer ---")
    composer = backtest.RandomStrategyComposer(df)
    
    print("--- 3. Running Walk-Forward Mining (Batch 50 - Vectorized) ---")
    try:
        results = composer.generate_and_test(n_strategies=50, train_ratio=0.7)
        if results.empty:
            print("No robust strategies found in this batch.")
        else:
            print(f"Found {len(results)} Robust Strategies!")
            print(results[['Description', 'Return', 'IS_Return', 'Score']].head())
            
            # Verify persistence
            composer.save_hall_of_fame(results)
            print("Saved to Hall of Fame.")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_alien_tech()
