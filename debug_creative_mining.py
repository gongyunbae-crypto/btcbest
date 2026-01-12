import pandas as pd
import backtest
import data_loader
import os

def test_creative_mining_logic():
    print("--- 1. Loading Data ---")
    df = data_loader.load_data()
    if df is None:
        print("Data not found locally. Fetching small sample...")
        df = data_loader.fetch_binance_futures_data("BTC/USDT", since_years=1)
    
    print(f"Data Loaded: {len(df)} records")

    print("--- 2. Initializing Composer ---")
    try:
        composer = backtest.RandomStrategyComposer(df)
        print("Composer Initialized.")
    except Exception as e:
        print(f"FAILED to initialize composer: {e}")
        return

    print("--- 3. Generating Strategies (Batch of 50) ---")
    try:
        results_df = composer.generate_and_test(n_strategies=50)
        
        if results_df.empty:
            print("Mining finished but returned NO valid strategies (all filtered?).")
        else:
            print(f"Success! Generated {len(results_df)} valid strategies.")
            print(results_df[['Description', 'Return', 'MDD', 'Score']].head())
            
            # Verify Hall of Fame Persistence
            print("\n--- 4. Testing Hall of Fame Persistence ---")
            
            # Clear existing for test
            if os.path.exists(composer.hof_file):
                os.remove(composer.hof_file)
                
            top_10 = composer.save_hall_of_fame(results_df)
            print(f"Saved {len(top_10)} to HoF.")
            
            # Reload
            loaded_hof = composer.load_hall_of_fame()
            print(f"Reloaded {len(loaded_hof)} from HoF.")
            
            if len(loaded_hof) == len(top_10):
                print("✅ Persistence Test Passed.")
            else:
                print("❌ Persistence Test Failed.")

    except Exception as e:
        print(f"❌ MINING FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_creative_mining_logic()
