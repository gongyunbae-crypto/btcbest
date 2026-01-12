import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def fetch_binance_data_auto(symbol="BTC/USDT", timeframe="5m", since_years=1):
    """
    Attempts to fetch Futures data, falls back to Spot if blocked.
    """
    print(f"Attempting to fetch Futures data for {symbol}...")
    df = fetch_binance_futures_data(symbol, timeframe, since_years)
    
    if df.empty:
        print("Futures data fetch failed/blocked. Attempting Spot data fallback...")
        exchange_spot = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        now = datetime.now()
        start_time = now - timedelta(days=365 * since_years)
        since_ms = int(start_time.timestamp() * 1000)
        
        all_ohlcv = []
        try:
            # Spot usually has more restrictive pagination but we'll try a basic fetch
            ohlcv = exchange_spot.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
            if ohlcv:
                all_ohlcv.extend(ohlcv)
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                print(f"Fallback successful: Fetched {len(df)} spot candles.")
        except Exception as e:
            print(f"Spot fallback also failed: {e}")
            
    return df

def fetch_binance_futures_data(symbol="BTC/USDT", timeframe="5m", since_years=1):

    """
    Fetches historical OHLCV data from Binance Futures via CCXT.
    Handles pagination to get long-term data.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Calculate start time
    now = datetime.now()
    if since_years > 0:
        start_time = now - timedelta(days=365 * since_years)
        since_ms = int(start_time.timestamp() * 1000)
    else:
         # Default fallback or manual override if needed
         since_ms = int((now - timedelta(days=365)).timestamp() * 1000)

    print(f"Fetching data for {symbol} ({timeframe}) starting from {start_time.isoformat()}...")

    all_ohlcv = []
    limit = 1000  # Binance limit per request
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update since_ms to the timestamp of the last candle + 1ms
            last_timestamp = ohlcv[-1][0]
            since_ms = last_timestamp + 1
            
            # Use data frame to check if we reached current time (roughly)
            # Or just rely on fetch_ohlcv returning empty list or partial list
            if len(ohlcv) < limit:
                break
                
            print(f"Fetched {len(all_ohlcv)} candles so far...")
            time.sleep(exchange.rateLimit / 1000.0) # Respect rate limit
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Sort and remove duplicates just in case
    df = df[~df.index.duplicated(keep='last')]
    df.sort_index(inplace=True)
    
    print(f"Total data fetched: {len(df)} rows.")
    return df

def save_data(df, filename="btc_futures_data_5m.csv"):
    df.to_csv(filename)
    print(f"Data saved to {filename}")

def load_data(filename="btc_futures_data_5m.csv"):
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
        # Ensure clean index
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        print(f"Data loaded from {filename} (Rows: {len(df)})")
        return df
    else:
        return None

def update_data(symbol, existing_df, timeframe="5m"):
    """
    Fetches only new candles since the last updated time in existing_df.
    Returns appended DataFrame.
    """
    if existing_df is None or existing_df.empty:
        return fetch_binance_futures_data(symbol, timeframe)
        
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # Last timestamp
    last_ts = existing_df.index[-1]
    since_ms = int(last_ts.timestamp() * 1000) + 1
    
    print(f"Checking for new data since {last_ts}...")
    
    new_candles = []
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms)
        if ohlcv:
            new_candles.extend(ohlcv)
            print(f"Found {len(new_candles)} new candles.")
    except Exception as e:
        print(f"Error updating data: {e}")
        return existing_df
        
    if not new_candles:
        print("No new data.")
        return existing_df
        
    # Create DF
    df_new = pd.DataFrame(new_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
    df_new.set_index('timestamp', inplace=True)
    
    # Append
    df_updated = pd.concat([existing_df, df_new])
    df_updated = df_updated[~df_updated.index.duplicated(keep='last')]
    df_updated.sort_index(inplace=True)
    
    save_data(df_updated)
    return df_updated
