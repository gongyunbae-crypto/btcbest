import pandas as pd
import vectorbt as vbt
import data_loader

def debug_broadcast_issue():
    print("Loading data...")
    df = data_loader.load_data()
    
    if df is None:
        print("No data found.")
        return

    print(f"Data shape: {df.shape}")
    print(f"Index type: {type(df.index)}")
    print(f"Has duplicates? {df.index.has_duplicates}")
    print(f"Is monotonic increasing? {df.index.is_monotonic_increasing}")
    
    print("\nAttempting VBT RSI run...")
    try:
        close = df['close'].astype(float)
        rsi = vbt.RSI.run(close, window=14)
        print("✅ VBT RSI Success!")
    except Exception as e:
        print(f"❌ VBT RSI Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_broadcast_issue()
