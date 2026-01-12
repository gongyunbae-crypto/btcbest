import pandas as pd
import numpy as np
import vectorbt as vbt

def get_indicators(df, rsi_window=14, ema_window=200):
    """
    Calculates technical indicators for the Strategy Mining Engine.
    Returns: rsi, ema, prev_high, prev_low
    """
    # Validate Data
    if df is None or df.empty:
        return pd.Series(), pd.Series(), pd.Series(), pd.Series()
        
    close_price = df['close'].astype(float)
    
    # RSI
    rsi = vbt.RSI.run(close_price, window=rsi_window).rsi
    
    # EMA (Trend Filter)
    ema = vbt.MA.run(close_price, window=ema_window, ewm=True).ma
    
    # Daily Levels (for Sweep)
    # Resample to Daily to get PDH/PDL
    df_d = df.resample('D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    })
    
    # Shift to get "Previous Day" levels available at "Today's Open"
    prev_high_d = df_d['high'].shift(1)
    prev_low_d = df_d['low'].shift(1)
    
    # Reindex to hourly to match original df
    # ffill() ensures that today's hourly candles see yesterday's high/low
    prev_high = prev_high_d.reindex(df.index).ffill()
    prev_low = prev_low_d.reindex(df.index).ffill()
    
    return rsi, ema, prev_high, prev_low

def apply_strategy_signals(df, indicators, sweep_depth_pct=0.005, rsi_threshold_low=30):
    """
    Generates entry/exit signals based on Liquidity Sweep, Trend, and RSI.
    """
    # Unpack indicators
    rsi, ema, prev_high, prev_low = indicators
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # --- Logic ---
    
    # Trend Filter
    is_bullish_trend = close > ema
    is_bearish_trend = close < ema
    
    # Liquidity Sweep Logic
    # Sweep Depth: Price went below PDL by X%?
    # Logic: Low < PDL * (1 - depth)
    swept_low = low < (prev_low * (1 - sweep_depth_pct))
    swept_high = high > (prev_high * (1 + sweep_depth_pct))
    
    # RSI Filter (Momentum/Reversal)
    # Long: RSI < Threshold (Oversold)
    # Short: RSI > (100 - Threshold) (Overbought)
    
    rsi_buy_cond = rsi < rsi_threshold_low
    rsi_sell_cond = rsi > (100 - rsi_threshold_low)
    
    # --- Entry Signals ---
    
    # Long: Bullish Trend + Swept Low + Close Reclaimed PDL + RSI Oversold
    long_signal = (
        swept_low & 
        (close > prev_low) & 
        is_bullish_trend & 
        rsi_buy_cond
    )
    
    # Short: Bearish Trend + Swept High + Close Loast PDH + RSI Overbought
    short_signal = (
        swept_high & 
        (close < prev_high) & 
        is_bearish_trend & 
        rsi_sell_cond
    )
    
    # Clean NaNs
    long_signal = long_signal.fillna(False)
    short_signal = short_signal.fillna(False)
    
    # --- Exit Signals (Trend Termination) ---
    long_exit = close < ema
    short_exit = close > ema
    
    long_exit = long_exit.fillna(False)
    short_exit = short_exit.fillna(False)
    
    return long_signal, short_signal, long_exit, short_exit

# Helper for legacy App UI (Live Analysis chart) to get levels easily
# We can keep get_daily_levels wrapper if needed, or update App to use get_indicators
# app.py (lines 82) calls: daily_levels = strategy.get_daily_levels(df)
# We MUST provide this function back for compatibility with the specific "Live Analysis" tab code I wrote earlier.

def get_daily_levels(df):
    """
    Legacy wrapper for UI visualization compatibility.
    Returns DataFrame with prev_high, prev_low, sma_200 columns.
    """
    rsi, ema, prev_high, prev_low = get_indicators(df)
    
    # Construct DataFrame
    daily_levels = pd.DataFrame(index=df.index)
    daily_levels['prev_high'] = prev_high
    daily_levels['prev_low'] = prev_low
    daily_levels['sma_200'] = ema # Using EMA as "SMA 200" for visual consistency or renamed
    
    
    return daily_levels

def apply_liquidity_sweep_strategy(df, sweep_depth_pct=0.005, rsi_threshold_low=30):
    """
    Legacy wrapper for App compatibility. 
    Returns DataFrame with 'Entry'/'Exit' columns or similar expected by legacy run_backtest.
    Actually, existing run_backtest likely expects 'long_signal' etc.
    Let's check how it was used:
    df_strat = strategy.apply_liquidity_sweep_strategy(...)
    backtest.run_backtest(df_strat, ...)
    """
    indicators = get_indicators(df)
    long, short, long_ex, short_ex = apply_strategy_signals(df, indicators, sweep_depth_pct, rsi_threshold_low)
    
    # Modify DF to include signals for legacy backtester
    df_out = df.copy()
    df_out['long_signal'] = long
    df_out['short_signal'] = short
    df_out['long_exit'] = long_ex
    df_out['short_exit'] = short_ex
    return df_out

# --- V3 Signal Factory ---

def get_expanded_indicators(df):
    """
    Calculates extended set of indicators for Creative Mining.
    """
    # Convert to pure float64 numpy arrays to prevent numba TypingError
    close = df['close'].values.astype('float64')
    high = df['high'].values.astype('float64')
    low = df['low'].values.astype('float64')
    volume = df['volume'].values.astype('float64')
    
    # 1. Base Indicators
    # 1.1 Fast RSI (Scalping)
    rsi_14 = vbt.RSI.run(close, window=14).rsi.to_numpy()
    rsi_9 = vbt.RSI.run(close, window=9).rsi.to_numpy()
    rsi_7 = vbt.RSI.run(close, window=7).rsi.to_numpy()
    
    ema_200 = vbt.MA.run(close, window=200, ewm=True).ma.to_numpy()
    ema_50 = vbt.MA.run(close, window=50, ewm=True).ma.to_numpy()
    ema_21 = vbt.MA.run(close, window=21, ewm=True).ma.to_numpy()
    ema_20 = vbt.MA.run(close, window=20, ewm=True).ma.to_numpy()
    ema_9 = vbt.MA.run(close, window=9, ewm=True).ma.to_numpy()
    ema_8 = vbt.MA.run(close, window=8, ewm=True).ma.to_numpy()
    
    # Volume SMA
    vol_sma = vbt.MA.run(volume, window=20).ma.to_numpy()
    
    # 2. Bollinger Bands
    bb_standard = vbt.BBANDS.run(close, window=20, alpha=2.0)
    bb_short = vbt.BBANDS.run(close, window=10, alpha=2.0)
    # Extract values (using to_numpy() for safety across versions)
    bb_upper = bb_standard.upper.to_numpy()
    bb_lower = bb_standard.lower.to_numpy()
    bb_middle = bb_standard.middle.to_numpy()
    bb_short_upper = bb_short.upper.to_numpy()
    bb_short_lower = bb_short.lower.to_numpy()
    
    # 3. MACD
    macd_std = vbt.MACD.run(close)
    macd_fast = vbt.MACD.run(close, fast_window=5, slow_window=13, signal_window=1)
    # Extract values
    macd_val = macd_std.macd.to_numpy()
    macd_sig = macd_std.signal.to_numpy()
    macd_fast_val = macd_fast.macd.to_numpy()
    macd_scalp_val = vbt.MACD.run(close, fast_window=8, slow_window=21, signal_window=5).macd.to_numpy()
    
    # 4. Stochastic
    stoch_std = vbt.STOCH.run(high, low, close) 
    stoch_fast = vbt.STOCH.run(high, low, close, k_window=5, d_window=3)
    # Extract values (Standardizing to k and d)
    stoch_k = stoch_std.percent_k.to_numpy()
    stoch_d = stoch_std.percent_d.to_numpy()
    stoch_fast_k = stoch_fast.percent_k.to_numpy()
    
    # 5. Volatility (ATR)
    atr = vbt.ATR.run(high, low, close).atr.to_numpy()
    
    # 5.1 VWAP (Volume Weighted Average Price) - Cumulative or Session
    # For crypto, often rolling or cumulative since day start. 
    # Here using cumulative sum for the loaded period as a simple baseline.
    cv = (close * df['volume']).cumsum()
    v = df['volume'].cumsum()
    vwap = cv / v
    
    # 5.2 OBV (On Balance Volume)
    obv = vbt.OBV.run(close, volume).obv.to_numpy()

    # 6. ADX (Trend Strength) - Manual Implementation (Using pd.Series wrappers for numpy inputs)
    s_high = pd.Series(high, index=df.index)
    s_low = pd.Series(low, index=df.index)
    s_close = pd.Series(close, index=df.index)
    
    plus_dm = s_high.diff()
    minus_dm = s_low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = s_high - s_low
    tr2 = (s_high - s_close.shift(1)).abs()
    tr3 = (s_low - s_close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_adx = tr.ewm(alpha=1/14, min_periods=14).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr_adx)
    minus_di = 100 * (minus_dm.abs().ewm(alpha=1/14, min_periods=14).mean() / atr_adx)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, min_periods=14).mean().to_numpy()
    plus_di_v = plus_di.to_numpy()
    minus_di_v = minus_di.to_numpy()

    # 7. CCI (Commodity Channel Index) - Manual Implementation
    tp = (s_high + s_low + s_close) / 3
    cci = ((tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std())).to_numpy()

    # --- Phase 20: Modern Indicators Expansion (User Request) ---

    # 8. Keltner Channels (KC)
    kc_upper = ema_20 + (2 * atr)
    kc_lower = ema_20 - (2 * atr)

    # 9. Williams %R
    hh_14 = s_high.rolling(14).max()
    ll_14 = s_low.rolling(14).min()
    williams_r = (((hh_14 - s_close) / (hh_14 - ll_14)) * -100).to_numpy()

    # 10. Awesome Oscillator (AO)
    median_price = (s_high + s_low) / 2
    ao = (median_price.rolling(5).mean() - median_price.rolling(34).mean()).to_numpy()

    # 11. Rate of Change (ROC)
    roc_9 = (((s_close - s_close.shift(9)) / s_close.shift(9)) * 100).to_numpy()

    # 12. Money Flow Index (MFI)
    raw_money_flow = tp * volume
    
    positive_flow = pd.Series(0.0, index=df.index)
    negative_flow = pd.Series(0.0, index=df.index)
    
    price_diff = tp.diff()
    positive_flow[price_diff > 0] = raw_money_flow[price_diff > 0]
    negative_flow[price_diff < 0] = raw_money_flow[price_diff < 0]
    
    mfi_ratio = positive_flow.rolling(14).sum() / negative_flow.rolling(14).sum()
    mfi = (100 - (100 / (1 + mfi_ratio))).to_numpy()

    # 13. Chande Momentum Oscillator (CMO)
    diff_p = s_close.diff()
    up_moves = diff_p.clip(lower=0)
    down_moves = diff_p.clip(upper=0).abs()
    
    sum_up = up_moves.rolling(9).sum()
    sum_down = down_moves.rolling(9).sum()
    cmo = (((sum_up - sum_down) / (sum_up + sum_down)) * 100).to_numpy()

    # 14. Ichimoku Cloud (Simplified)
    tenkan_sen = ((s_high.rolling(9).max() + s_low.rolling(9).min()) / 2).to_numpy()
    kijun_sen = ((s_high.rolling(26).max() + s_low.rolling(26).min()) / 2).to_numpy()
    # Span A, Span B, Chikou require shifting, keeping simple for now (Tenkan/Kijun Cross)

    # 15. SuperTrend (Proxy)
    psar_long = (s_high.rolling(22).max() - (3 * atr)).to_numpy()
    psar_short = (s_low.rolling(22).min() + (3 * atr)).to_numpy()
    # Logic will be in SignalRepository (Price > Lower vs Price < Upper)

    # PSAR / SuperTrend Proxy already calculated in Step 15 above.


    # --- Phase 21: Mega Expansion (User Request - Double Indicators) ---

    # 17. Donchian Channels (20)
    donchian_upper = s_high.rolling(20).max().to_numpy()
    donchian_lower = s_low.rolling(20).min().to_numpy()
    donchian_mid = (donchian_upper + donchian_lower) / 2

    # 18. Vortex Indicator (VI)
    tr_1 = pd.concat([s_high - s_low, (s_high - s_close.shift(1)).abs(), (s_low - s_close.shift(1)).abs()], axis=1).max(axis=1)
    vm_plus = (s_high - s_low.shift(1)).abs()
    vm_minus = (s_low - s_high.shift(1)).abs()
    
    tr14 = tr_1.rolling(14).sum()
    vi_plus = (vm_plus.rolling(14).sum() / (tr14 + 1e-8)).to_numpy()
    vi_minus = (vm_minus.rolling(14).sum() / (tr14 + 1e-8)).to_numpy()

    # 19. TRIX (Triple Exponential Average)
    ema1 = s_close.ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix = (ema3.pct_change() * 100).to_numpy()

    # 20. Force Index
    fi = ((s_close.diff(1) * df['volume']).ewm(span=13, adjust=False).mean()).to_numpy()

    # 21. Ease of Movement (EOM)
    # Distance moved / Box Ratio
    # Box Ratio = (Vol / Scale) / (High - Low)
    # Scale often 10000 or 1000000 to make numbers readable.
    dist_moved = ((s_high + s_low) / 2) - ((s_high.shift(1) + s_low.shift(1)) / 2)
    box_ratio = (df['volume'] / 100000000) / ((s_high - s_low) + 0.00001) # Avoid div by zero
    eom = (dist_moved / box_ratio).rolling(14).mean().to_numpy()

    # 22. Chaikin Money Flow (CMF)
    mfm = ((s_close - s_low) - (s_high - s_close)) / ((s_high - s_low) + 1e-8)
    mfv = mfm * df['volume']
    cmf = (mfv.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-8)).to_numpy()

    # 23. VWMA (Volume Weighted Moving Average)
    vwma_20 = ((s_close * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-8)).to_numpy()

    # 24. Hull Moving Average (HMA)
    # WMA likely needed. Pandas doesn't have WMA built-in easily.
    # Using EMA approximation for speed or implementing WMA helper?
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    # We will use Weighted Mean manually.
    def weighted_mean(s, w):
        return pd.Series(np.average(s, weights=w, axis=0), index=[s.index[-1]])
    
    # Actually, for speed in "Mining Engine", let's use a faster proxy:
    # 2*EMA(n/2) - EMA(n) -> EMA(sqrt(n))
    # It behaves very similarly to HMA (responsive).
    hma_n = 9
    ema_half = s_close.ewm(span=int(hma_n/2), adjust=False).mean()
    ema_full = s_close.ewm(span=hma_n, adjust=False).mean()
    raw_hma = 2 * ema_half - ema_full
    hma_9 = raw_hma.ewm(span=int(np.sqrt(hma_n)), adjust=False).mean().to_numpy()

    # 25. Connors RSI (CRSI)
    # 1. RSI(3)
    rsi_3 = vbt.RSI.run(close, window=3).rsi.to_numpy()
    rsi_roc = vbt.RSI.run(s_close.pct_change().fillna(0).to_numpy(), window=3).rsi.to_numpy()
    connors_rsi_proxy = (rsi_3 + rsi_roc) / 2

    # 26. Aroon
    # Days since N-day High
    # Rolling apply or simple argmax?
    # Pandas rolling.apply with argmax is slow.
    # Alternative: VectorBT might filter?
    # Let's implement manually with 'rolling' and checking offset? No.
    # Let's use simpler High/Low proximity.
    # Aroon Up = (N - (Index of High)) / N * 100
    # Rolling Max Index is available in pandas > 1.?
    # For now, let's use a simpler proxy: Donchian Position.
    # Aroon Like = (Close - LowestLow) / (HighestHigh - LowestLow) * 100 (This is Stochastic!)
    # True Aroon measures TIME.
    # Let's Skip Aroon (Time-based high complexity) and use 'Ultimate Oscillator'.

    # 26. Ultimate Oscillator
    prev_close = s_close.shift(1)
    true_low = pd.concat([s_low, prev_close], axis=1).min(axis=1)
    true_high = pd.concat([s_high, prev_close], axis=1).max(axis=1)
    bp = s_close - true_low
    tr_u = true_high - true_low
    
    avg7 = bp.rolling(7).sum() / (tr_u.rolling(7).sum() + 1e-8)
    avg14 = bp.rolling(14).sum() / (tr_u.rolling(14).sum() + 1e-8)
    avg28 = bp.rolling(28).sum() / (tr_u.rolling(28).sum() + 1e-8)
    ultimate = ((4 * avg7 + 2 * avg14 + avg28) / 7 * 100).to_numpy()

    # 27. KST (Know Sure Thing)
    def roc_ma(n, ma): 
        return (s_close.diff(n)/s_close.shift(n)).rolling(ma).mean()
    
    kst = ((roc_ma(10, 10) * 1 + roc_ma(15, 10) * 2 + roc_ma(20, 10) * 3 + roc_ma(30, 15) * 4) * 100)
    kst_v = kst.to_numpy()
    kst_signal_v = kst.rolling(9).mean().to_numpy()

    return {
        'close': close, 'high': high, 'low': low, 'volume': volume,
        'rsi': rsi_14, 'rsi_9': rsi_9, 'rsi_7': rsi_7,
        'ema_200': ema_200, 'ema_50': ema_50, 'ema_21': ema_21, 'ema_20': ema_20, 'ema_9': ema_9, 'ema_8': ema_8,
        'vol_sma': vol_sma,
        'bb_up': bb_upper, 'bb_low': bb_lower,
        'bb_up_short': bb_short_upper, 'bb_low_short': bb_short_lower,
        'macd': macd_val, 'macd_sig': macd_sig,
        'macd_fast': macd_fast_val, 'macd_scalp': macd_scalp_val,
        'stoch_k': stoch_k, 'stoch_d': stoch_d,
        'stoch_fast_k': stoch_fast_k,
        'vwap': vwap.to_numpy() if hasattr(vwap, "to_numpy") else vwap, 
        'obv': obv,
        'atr': atr, 'adx': adx, 'cci': cci,
        'kc_upper': kc_upper, 'kc_lower': kc_lower,
        'williams_r': williams_r, 'ao': ao, 'roc': roc_9,
        'mfi': mfi, 'cmo': cmo, 'tenkan': tenkan_sen, 'kijun': kijun_sen,
        'psar_long': psar_long, 'psar_short': psar_short,
        'donchian_upper': donchian_upper, 'donchian_lower': donchian_lower,
        'vi_plus': vi_plus, 'vi_minus': vi_minus,
        'trix': trix, 'fi': fi, 'eom': eom, 'cmf': cmf,
        'vwma_20': vwma_20, 'hma_9': hma_9, 'crsi': connors_rsi_proxy,
        'ultimate': ultimate, 'kst': kst_v, 'kst_sig': kst_signal_v
    }

class SignalRepository:
    """
    Generates atomic boolean signals from indicators.
    """
    def __init__(self, indicators):
        self.ind = indicators
        self.ind = indicators
        self.signals = {}
        self.SYMMETRY_MAP = {
            # Trend
            'Trend_Bullish_200': 'Trend_Bearish_200',
            'Trend_Bearish_200': 'Trend_Bullish_200',
            # Crosses
            'EMA_50_GT_EMA_200': 'EMA_50_LT_EMA_200',
            'EMA_50_LT_EMA_200': 'EMA_50_GT_EMA_200',
            # RSI
            'RSI_Oversold_30': 'RSI_Overbought_70',
            'RSI_Overbought_70': 'RSI_Oversold_30',
            'RSI_9_Oversold_30': 'RSI_9_Overbought_70',
            'RSI_9_Overbought_70': 'RSI_9_Oversold_30',
            'RSI_7_Oversold_25': 'RSI_7_Overbought_75',
            'RSI_7_Overbought_75': 'RSI_7_Oversold_25',
            # BB
            'Price_Below_BB_Lower': 'Price_Above_BB_Upper',
            'Price_Above_BB_Upper': 'Price_Below_BB_Lower',
            'Price_Below_BB_Low_Short': 'Price_Above_BB_Up_Short',
            'Price_Above_BB_Up_Short': 'Price_Below_BB_Low_Short',
            # MACD
            'MACD_Bullish_Cross': 'MACD_Bearish_Cross',
            'MACD_Bearish_Cross': 'MACD_Bullish_Cross',
            'MACD_Scalp_Bull_Cross': 'MACD_Scalp_Bear_Cross',
            'MACD_Scalp_Bear_Cross': 'MACD_Scalp_Bull_Cross',
            'MACD_Fast_Bull_Signal': 'MACD_Fast_Bear_Signal',
            'MACD_Fast_Bear_Signal': 'MACD_Fast_Bull_Signal',
            # EMA Crosses
            'EMA_9_Cross_20_Up': 'EMA_9_Cross_20_Down',
            'EMA_9_Cross_20_Down': 'EMA_9_Cross_20_Up',
            'EMA_8_Cross_21_Up': 'EMA_8_Cross_21_Down',
            'EMA_8_Cross_21_Down': 'EMA_8_Cross_21_Up',
            # AO
            'AO_Bullish_Zero_Cross': 'AO_Bearish_Zero_Cross',
            'AO_Bearish_Zero_Cross': 'AO_Bullish_Zero_Cross',
            'AO_Trending_Up': 'AO_Trending_Down', # Need to add AO_Trending_Down below
            # ROC
            'ROC_Positive': 'ROC_Negative',
            'ROC_Negative': 'ROC_Positive',
            # MFI
            'MFI_Oversold_20': 'MFI_Overbought_80',
            'MFI_Overbought_80': 'MFI_Oversold_20',
            # CMF
            'CMF_Buying_Pressure': 'CMF_Selling_Pressure',
            'CMF_Selling_Pressure': 'CMF_Buying_Pressure',
            # VWAP
            'Price_GT_VWAP': 'Price_LT_VWAP',
            'Price_LT_VWAP': 'Price_GT_VWAP',
            # OBV
            'OBV_Rising': 'OBV_Falling',
            'OBV_Falling': 'OBV_Rising',
            # Stoch
            'Stoch_Oversold_20': 'Stoch_Overbought_80',
            'Stoch_Overbought_80': 'Stoch_Oversold_20',
            'Stoch_Fast_Oversold_20': 'Stoch_Fast_Overbought_80',
            'Stoch_Fast_Overbought_80': 'Stoch_Fast_Oversold_20',
             # Ichimoku
            'Ichimoku_TK_Cross_Bull': 'Ichimoku_TK_Cross_Bear',
            'Ichimoku_TK_Cross_Bear': 'Ichimoku_TK_Cross_Bull',
            # Vortex
            'Vortex_Bullish_Cross': 'Vortex_Bearish_Cross',
            'Vortex_Bearish_Cross': 'Vortex_Bullish_Cross',
            # Hull
            'HMA_Trending_Up': 'HMA_Trending_Down',
            'HMA_Trending_Down': 'HMA_Trending_Up',
        }
        self._generate_atomic_signals()
        
    def _generate_atomic_signals(self):
        ind = self.ind
        close = ind['close']
        
        # --- Atomic Signals ---
        
        # Trend
        self.signals['Trend_Bullish_200'] = close > ind['ema_200']
        self.signals['Trend_Bearish_200'] = close < ind['ema_200']
        self.signals['Trend_Bullish_50'] = close > ind['ema_50']
        
        # Crosses (Golden/Death)
        self.signals['EMA_50_GT_EMA_200'] = ind['ema_50'] > ind['ema_200']
        self.signals['EMA_50_LT_EMA_200'] = ind['ema_50'] < ind['ema_200']
        
        # RSI
        self.signals['RSI_Oversold_30'] = ind['rsi'] < 30
        self.signals['RSI_Overbought_70'] = ind['rsi'] > 70
        
        # Bollinger Bands
        self.signals['Price_Below_BB_Lower'] = close < ind['bb_low']
        self.signals['Price_Above_BB_Upper'] = close > ind['bb_up']
        
        # MACD
        self.signals['MACD_Bullish_Cross'] = (ind['macd'] > ind['macd_sig']) & (ind['macd'].shift(1) < ind['macd_sig'].shift(1))
        self.signals['MACD_Bearish_Cross'] = (ind['macd'] < ind['macd_sig']) & (ind['macd'].shift(1) > ind['macd_sig'].shift(1))
        
        # Stochastic
        self.signals['Stoch_Oversold_20'] = ind['stoch_k'] < 20
        self.signals['Stoch_Overbought_80'] = ind['stoch_k'] > 80

        # ADX (Trend Strength)
        self.signals['ADX_Strong_Trend_25'] = ind['adx'] > 25
        self.signals['ADX_Weak_Trend_20'] = ind['adx'] < 20

        # CCI
        self.signals['CCI_Oversold_M100'] = ind['cci'] < -100
        self.signals['CCI_Overbought_100'] = ind['cci'] > 100
        
        # --- Scalping Specific Signals (Added Phase 12) ---
        
        # Fast EMA Crossovers
        # 9 cross 20
        self.signals['EMA_9_Cross_20_Up'] = (ind['ema_9'] > ind['ema_20']) & (ind['ema_9'].shift(1) < ind['ema_20'].shift(1))
        self.signals['EMA_9_Cross_20_Down'] = (ind['ema_9'] < ind['ema_20']) & (ind['ema_9'].shift(1) > ind['ema_20'].shift(1))
        
        # 20 cross 50
        self.signals['EMA_20_Cross_50_Up'] = (ind['ema_20'] > ind['ema_50']) & (ind['ema_20'].shift(1) < ind['ema_50'].shift(1))
        self.signals['EMA_20_Cross_50_Down'] = (ind['ema_20'] < ind['ema_50']) & (ind['ema_20'].shift(1) > ind['ema_50'].shift(1))
        
        # Volume Spike (1.5x of SMA 20)
        self.signals['Volume_High'] = ind['volume'] > (ind['vol_sma'] * 1.5)
        
        # Volume Pump (1.2x of SMA 20) - For MACD+Vol Strategy
        self.signals['Volume_Pump_1.2'] = ind['volume'] > (ind['vol_sma'] * 1.2)
        
        # --- Modern Indicators (Phase 20) ---

        # 1. Keltner Channels
        self.signals['KC_Breakout_Upper'] = close > ind['kc_upper']
        self.signals['KC_Breakout_Lower'] = close < ind['kc_lower']
        
        # 2. Williams %R (-20 is OB, -80 is OS)
        self.signals['Williams_Overbought_M20'] = ind['williams_r'] > -20
        self.signals['Williams_Oversold_M80'] = ind['williams_r'] < -80
        
        # 3. Awesome Oscillator
        self.signals['AO_Bullish_Zero_Cross'] = (ind['ao'] > 0) & (ind['ao'].shift(1) < 0)
        self.signals['AO_Bearish_Zero_Cross'] = (ind['ao'] < 0) & (ind['ao'].shift(1) > 0)
        self.signals['AO_Trending_Up'] = ind['ao'] > ind['ao'].shift(1)
        self.signals['AO_Trending_Down'] = ind['ao'] < ind['ao'].shift(1)
        
        # 4. ROC
        self.signals['ROC_Positive'] = ind['roc'] > 0
        self.signals['ROC_Negative'] = ind['roc'] < 0
        self.signals['ROC_Surge_2pct'] = ind['roc'] > 2.0
        
        # 5. MFI
        self.signals['MFI_Overbought_80'] = ind['mfi'] > 80
        self.signals['MFI_Oversold_20'] = ind['mfi'] < 20
        
        # 6. CMO
        self.signals['CMO_Overbought_50'] = ind['cmo'] > 50
        self.signals['CMO_Oversold_M50'] = ind['cmo'] < -50
        
        # 7. Ichimoku
        self.signals['Ichimoku_TK_Cross_Bull'] = (ind['tenkan'] > ind['kijun']) & (ind['tenkan'].shift(1) <= ind['kijun'].shift(1))
        self.signals['Ichimoku_TK_Cross_Bear'] = (ind['tenkan'] < ind['kijun']) & (ind['tenkan'].shift(1) >= ind['kijun'].shift(1))
        self.signals['Ichimoku_Price_Above_Cloud_Proxy'] = close > ind['kijun'] # Simplified trend
        
        # 8. PSAR / SuperTrend Proxy
        # Long if Price > PSAR Down (Stop is below price)
        # Short if Price < PSAR Up (Stop is above price)
        # Here psar_long is the "Stop Level" for Longs (Below price)
        self.signals['PSAR_Trend_Bullish'] = close > ind['psar_long'] 
        self.signals['PSAR_Trend_Bearish'] = close < ind['psar_short']

        # --- Mega Expansion Signals (Phase 21) ---

        # 9. Donchian Channels
        self.signals['Donchian_Breakout_Upper'] = close > ind['donchian_upper'].shift(1)
        self.signals['Donchian_Breakout_Lower'] = close < ind['donchian_lower'].shift(1)

        # 10. Vortex Indicator
        self.signals['Vortex_Bullish_Cross'] = (ind['vi_plus'] > ind['vi_minus']) & (ind['vi_plus'].shift(1) <= ind['vi_minus'].shift(1))
        self.signals['Vortex_Bearish_Cross'] = (ind['vi_plus'] < ind['vi_minus']) & (ind['vi_plus'].shift(1) >= ind['vi_minus'].shift(1))

        # 11. TRIX
        self.signals['TRIX_Bullish_Zero'] = (ind['trix'] > 0) & (ind['trix'].shift(1) < 0)
        self.signals['TRIX_Bearish_Zero'] = (ind['trix'] < 0) & (ind['trix'].shift(1) > 0)

        # 12. KST
        self.signals['KST_Bullish_Cross'] = (ind['kst'] > ind['kst_signal']) & (ind['kst'].shift(1) <= ind['kst_signal'].shift(1))
        self.signals['KST_Bearish_Cross'] = (ind['kst'] < ind['kst_signal']) & (ind['kst'].shift(1) >= ind['kst_signal'].shift(1))

        # 13. Force Index
        self.signals['Force_Bullish_Zero'] = ind['fi'] > 0
        self.signals['Force_Bearish_Zero'] = ind['fi'] < 0
        
        # 14. Ease of Movement
        self.signals['EOM_Easy_Rise'] = ind['eom'] > 0.000001
        self.signals['EOM_Easy_Fall'] = ind['eom'] < -0.000001

        # 15. Chaikin Money Flow
        self.signals['CMF_Buying_Pressure'] = ind['cmf'] > 0.05
        self.signals['CMF_Selling_Pressure'] = ind['cmf'] < -0.05

        # 16. VWMA vs SMA
        self.signals['VWMA_GT_SMA'] = ind['vwma_20'] > ind['ema_20'] # vs EMA20
        self.signals['VWMA_LT_SMA'] = ind['vwma_20'] < ind['ema_20']

        # 17. Hull Moving Average Trend
        self.signals['HMA_Trending_Up'] = ind['hma_9'] > ind['hma_9'].shift(1)
        self.signals['HMA_Trending_Down'] = ind['hma_9'] < ind['hma_9'].shift(1)

        # 18. Connors RSI (Scalping)
        # 30/70 levels for CRSI? Usually 10/90 or 5/95 for strict.
        self.signals['CRSI_Oversold_15'] = ind['crsi'] < 15
        self.signals['CRSI_Overbought_85'] = ind['crsi'] > 85

        # 19. Ultimate Oscillator
        # 30/70 levels
        self.signals['UO_Oversold_30'] = ind['ultimate'] < 30
        self.signals['UO_Overbought_70'] = ind['ultimate'] > 70
        
        # --- Phase 22: Scalper/Short-Term Expansion (2025-2026 Trends) ---
        
        # 1. EMA 8/21 Cross
        self.signals['EMA_8_Cross_21_Up'] = (ind['ema_8'] > ind['ema_21']) & (ind['ema_8'].shift(1) <= ind['ema_21'].shift(1))
        self.signals['EMA_8_Cross_21_Down'] = (ind['ema_8'] < ind['ema_21']) & (ind['ema_8'].shift(1) >= ind['ema_21'].shift(1))
        
        # 2. Fast RSI (9)
        self.signals['RSI_9_Oversold_30'] = ind['rsi_9'] < 30
        self.signals['RSI_9_Overbought_70'] = ind['rsi_9'] > 70
        
        # 3. Fast RSI (7)
        self.signals['RSI_7_Oversold_25'] = ind['rsi_7'] < 25
        self.signals['RSI_7_Overbought_75'] = ind['rsi_7'] > 75
        
        # 4. Scalp MACD (8, 21, 5)
        self.signals['MACD_Scalp_Bull_Cross'] = (ind['macd_scalp'] > ind['macd_scalp_sig']) & (ind['macd_scalp'].shift(1) <= ind['macd_scalp_sig'].shift(1))
        self.signals['MACD_Scalp_Bear_Cross'] = (ind['macd_scalp'] < ind['macd_scalp_sig']) & (ind['macd_scalp'].shift(1) >= ind['macd_scalp_sig'].shift(1))
        
        # 5. Fast MACD (5, 13, 1)
        self.signals['MACD_Fast_Bull_Signal'] = ind['macd_fast'] > 0 # Zero-cross
        self.signals['MACD_Fast_Bear_Signal'] = ind['macd_fast'] < 0
        
        # 6. BB Short (10, 2)
        self.signals['Price_Below_BB_Low_Short'] = close < ind['bb_low_short']
        self.signals['Price_Above_BB_Up_Short'] = close > ind['bb_up_short']
        
        # 7. VWAP Logic
        self.signals['Price_GT_VWAP'] = close > ind['vwap']
        self.signals['Price_LT_VWAP'] = close < ind['vwap']
        
        # 8. OBV Logic
        self.signals['OBV_Rising'] = ind['obv'] > ind['obv'].shift(1)
        self.signals['OBV_Falling'] = ind['obv'] < ind['obv'].shift(1)
        
        # 9. Fast Stochastic (5, 3, 3)
        self.signals['Stoch_Fast_Oversold_20'] = ind['stoch_fast_k'] < 20
        self.signals['Stoch_Fast_Overbought_80'] = ind['stoch_fast_k'] > 80
        
    def get_random_signal_pair(self):
        """Returns two random boolean signals and a logic operator."""
        import random
        keys = list(self.signals.keys())
        s1 = random.choice(keys)
        s2 = random.choice(keys)
        op = random.choice(['AND', 'OR'])
        return s1, s2, op

    def get_symmetric_logic(self, s1, s2, op):
        """Returns the symmetric version of the logic for the opposite direction."""
        sym_s1 = self.SYMMETRY_MAP.get(s1, s1) # Fallback to same if no map
        sym_s2 = self.SYMMETRY_MAP.get(s2, s2)
        return sym_s1, sym_s2, op

    def evaluate_combined_signal(self, s1_key, s2_key, operator):
        """Evaluates (Signal1 OP Signal2)."""
        series1 = self.signals[s1_key]
        series2 = self.signals[s2_key]
        
        if operator == 'AND':
            return series1 & series2
        else: # OR
            return series1 | series2

    def get_signal_matrix(self):
        """
        Returns (columns, matrix_values) for vectorized operations.
        columns: List of signal names (keys)
        values: Bool numpy array of shape (Time, N_Signals)
        """
        keys = list(self.signals.keys())
        # Ensure consistent order
        keys.sort()
        
        # Stack into matrix
        # self.signals is dict of pd.Series
        # We can use pd.DataFrame(self.signals) but let's be explicit with keys
        df_sigs = pd.DataFrame({k: self.signals[k] for k in keys})
        
        return keys, df_sigs.values, df_sigs.index

# --- Internet Strategy Presets (Added Phase 15) ---

def apply_golden_cross_strategy(df):
    """
    Classic Golden Cross: Long when SMA 50 > SMA 200.
    Exit when SMA 50 < SMA 200.
    """
    ind = get_expanded_indicators(df)
    
    # Logic
    # Crossover: 50 > 200 (Golden)
    # Crossunder: 50 < 200 (Death)
    
    long_signal = (ind['ema_50'] > ind['ema_200']) & (ind['ema_50'].shift(1) <= ind['ema_200'].shift(1))
    short_signal = (ind['ema_50'] < ind['ema_200']) & (ind['ema_50'].shift(1) >= ind['ema_200'].shift(1))
    
    # Exits (Reverse of entry for trend following)
    long_exit = short_signal
    short_exit = long_signal
    
    # Fillna
    long_signal = long_signal.fillna(False)
    short_signal = short_signal.fillna(False)
    long_exit = long_exit.fillna(False)
    short_exit = short_exit.fillna(False)
    
    # Create DF
    df_out = df.copy()
    df_out['long_signal'] = long_signal
    df_out['short_signal'] = short_signal
    df_out['long_exit'] = long_exit
    df_out['short_exit'] = short_exit
    
    return df_out

def apply_rsi_bollinger_strategy(df):
    """
    Reversal Strategy:
    Long: Price < Lower BB AND RSI < 30
    Short: Price > Upper BB AND RSI > 70
    Exit: Touch Basis (Middle Band)
    """
    ind = get_expanded_indicators(df)
    
    close = ind['close']
    
    # Entry
    long_signal = (close < ind['bb_lower']) & (ind['rsi'] < 30)
    short_signal = (close > ind['bb_upper']) & (ind['rsi'] > 70)
    
    # Exit (Mean Reversion)
    # Long Exit: Price > Basis
    # Short Exit: Price < Basis
    long_exit = close > ind['ema_20'] # Basis is SMA 20 usually, here using EMA 20 as proxy or re-calc basis
    short_exit = close < ind['ema_20']
    
    # Fillna
    long_signal = long_signal.fillna(False)
    short_signal = short_signal.fillna(False)
    long_exit = long_exit.fillna(False)
    short_exit = short_exit.fillna(False)
    
    df_out = df.copy()
    df_out['long_signal'] = long_signal
    df_out['short_signal'] = short_signal
    df_out['long_exit'] = long_exit
    df_out['short_exit'] = short_exit
    
    return df_out

def apply_scalping_crossover_strategy(df):
    """
    Scalping: EMA 9 Cross EMA 20
    """
    ind = get_expanded_indicators(df)
    
    long_signal = (ind['ema_9'] > ind['ema_20']) & (ind['ema_9'].shift(1) <= ind['ema_20'].shift(1))
    short_signal = (ind['ema_9'] < ind['ema_20']) & (ind['ema_9'].shift(1) >= ind['ema_20'].shift(1))
    
    long_exit = short_signal
    short_exit = long_signal
    
    long_signal = long_signal.fillna(False)
    short_signal = short_signal.fillna(False)
    long_exit = long_exit.fillna(False)
    short_exit = short_exit.fillna(False)
    
    df_out = df.copy()
    df_out['long_signal'] = long_signal
    df_out['short_signal'] = short_signal
    df_out['long_exit'] = long_exit
    df_out['short_exit'] = short_exit
    
    return df_out

def apply_macd_volume_strategy(df):
    """
    MACD + Volume Strategy (User Request)
    Long: MACD Crossover Up AND Volume > 1.2 * SMA20
    Short: MACD Crossover Down AND Volume > 1.2 * SMA20
    Exit: Reverse Signal
    """
    ind = get_expanded_indicators(df)
    
    # Logic
    # MACD Crossover (Bullish)
    macd_bull = (ind['macd_line'] > ind['macd_signal']) & (ind['macd_line'].shift(1) <= ind['macd_signal'].shift(1))
    
    # MACD Crossover (Bearish)
    macd_bear = (ind['macd_line'] < ind['macd_signal']) & (ind['macd_line'].shift(1) >= ind['macd_signal'].shift(1))
    
    # Volume Check
    vol_valid = ind['volume'] > (ind['vol_sma'] * 1.2)
    
    # Combine
    long_signal = macd_bull & vol_valid
    short_signal = macd_bear & vol_valid
    
    # Exits: Reverse Signal (Stop and Reverse)
    # Strategy says: "Exit when opposite signal occurs"
    # But usually Stop and Reverse implies existing position acts as exit?
    # Simple Logic: Exit on Opposite Entry Signal
    
    long_exit = short_signal
    short_exit = long_signal
    
    # Fillna
    long_signal = long_signal.fillna(False)
    short_signal = short_signal.fillna(False)
    long_exit = long_exit.fillna(False)
    short_exit = short_exit.fillna(False)
    
    df_out = df.copy()
    df_out['long_signal'] = long_signal
    df_out['short_signal'] = short_signal
    df_out['long_exit'] = long_exit
    df_out['short_exit'] = short_exit
    
    return df_out
