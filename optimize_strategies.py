import pandas as pd
import numpy as np
import strategy
import data_loader

def run_advanced_backtest(df, long_signals, short_signals, strat_name, mode='baseline', leverage=1, base_sl=0.03, base_tp=0.1):
    """
    Custom Event-Driven Backtester supporting DCA and Partial TP.
    modes: 'baseline', 'dca', 'partial_tp'
    """
    
    balance = 10000.0
    position_size = 0.0
    avg_entry_price = 0.0
    entry_fee = 0.0004
    
    in_position = False
    partial_taken = False
    dca_filled = False
    
    trades = []
    equity_curve = [balance]
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    times = df.index
    
    long_arr = long_signals.values
    
    print(f"\n--- Simulation: {strat_name} | Mode: {mode.upper()} ---")
    
    for i in range(len(df)):
        current_close = closes[i]
        current_high = highs[i]
        current_low = lows[i]
        
        # 1. Check Exit Conditions
        if in_position:
            # --- BASELINE ---
            if mode == 'baseline':
                exit_sl = avg_entry_price * (1 - base_sl)
                exit_tp = avg_entry_price * (1 + base_tp)
                
                if current_low <= exit_sl:
                    pnl = position_size * -base_sl
                    balance += pnl - (position_size * entry_fee)
                    in_position = False
                    trades.append({'pnl': pnl, 'ret': -base_sl})
                elif current_high >= exit_tp:
                    pnl = position_size * base_tp
                    balance += pnl - (position_size * entry_fee)
                    in_position = False
                    trades.append({'pnl': pnl, 'ret': base_tp})
            
            # --- DCA ---
            elif mode == 'dca':
                dca_price = avg_entry_price * 0.98
                
                # Check DCA Fill
                if not dca_filled and current_low <= dca_price:
                    # Fill DCA
                    dca_amt = 10000.0 * leverage * 0.5
                    # New Avg Price logic
                    # Val1 = Size. Val2 = Size.
                    # Qty1 = Size/P1. Qty2 = Size/P2.
                    # Pav = (Val1+Val2)/(Qty1+Qty2)
                    qty1 = position_size / avg_entry_price
                    qty2 = dca_amt / dca_price
                    position_size += dca_amt
                    avg_entry_price = position_size / (qty1 + qty2)
                    dca_filled = True
                
                # Check Exits with potentially new avg price
                # SL is -5% of Avg
                exit_sl = avg_entry_price * 0.95
                exit_tp = avg_entry_price * (1 + base_tp)
                
                if current_low <= exit_sl:
                    pnl = position_size * -0.05
                    balance += pnl - (position_size * entry_fee)
                    in_position = False
                    trades.append({'pnl': pnl, 'ret': -0.05})
                elif current_high >= exit_tp:
                    pnl = position_size * base_tp
                    balance += pnl - (position_size * entry_fee)
                    in_position = False
                    trades.append({'pnl': pnl, 'ret': base_tp})
            
            # --- PARTIAL TP ---
            elif mode == 'partial_tp':
                tp1_price = avg_entry_price * 1.03
                
                # Check Partial Fill
                if not partial_taken and current_high >= tp1_price:
                    sell_amt = position_size * 0.5
                    realized = sell_amt * 0.03
                    balance += realized - (sell_amt * entry_fee)
                    position_size -= sell_amt
                    partial_taken = True
                
                # Check Exits
                # SL: If partial taken, BE (Entry). Else Base SL.
                exit_sl = avg_entry_price if partial_taken else (avg_entry_price * (1 - base_sl))
                exit_tp = avg_entry_price * (1 + base_tp)
                
                if current_low <= exit_sl:
                    # Stopped
                    if partial_taken:
                        balance -= position_size * entry_fee # Break even stop
                        trades.append({'pnl': 0, 'ret': 0})
                    else:
                        pnl = position_size * -base_sl
                        balance += pnl - (position_size * entry_fee)
                        trades.append({'pnl': pnl, 'ret': -base_sl})
                    in_position = False
                elif current_high >= exit_tp:
                    # Full Target
                    pnl = position_size * base_tp
                    balance += pnl - (position_size * entry_fee)
                    in_position = False
                    trades.append({'pnl': pnl, 'ret': base_tp})

        # 2. Check Entry
        if not in_position and long_arr[i]:
            if mode == 'dca':
                position_size = 10000.0 * leverage * 0.5
            else:
                position_size = 10000.0 * leverage
            
            # Entry at Close standard VBT assumption (Next Open is more realistic but VBT backtest used Close)
            avg_entry_price = current_close 
            in_position = True
            partial_taken = False
            dca_filled = False
            balance -= position_size * entry_fee
        
        equity_curve.append(balance)

    # Metrics
    equity = np.array(equity_curve)
    ret_pct = (balance - 10000.0)/10000.0 * 100
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak)/peak
    mdd = np.min(dd) * 100
    wins = len([t for t in trades if t['ret'] > 0])
    cnt = len(trades)
    wr = (wins/cnt*100) if cnt>0 else 0
    
    return {'Return': ret_pct, 'MDD': mdd, 'WinRate': wr, 'Trades': cnt}

# --- Exec ---
df = data_loader.load_data()
ind = strategy.get_expanded_indicators(df)
repo = strategy.SignalRepository(ind)

strats = [
    {'name': "Aggressive (Vortex+KST)", 's1': "Vortex_Bullish_Cross", 's2': "KST_Bullish_Cross", 'lev': 2, 'sl': 0.0285, 'tp': 0.1},
    {'name': "Conservative (EMA+Ichi)", 's1': "EMA_9_Cross_20_Down", 's2': "Ichimoku_TK_Cross_Bull", 'lev': 1, 'sl': 0.0387, 'tp': 0.1},
    {'name': "Reversal (MACD+KC)", 's1': "MACD_Bullish_Cross", 's2': "KC_Breakout_Lower", 'lev': 2, 'sl': 0.0456, 'tp': 0.1}
]

print(f"{'Strategy':<25} | {'Mode':<10} | {'Return':<8} | {'MDD':<8} | {'WinRate':<8}")
print("-" * 75)

for s in strats:
    l_sig = repo.evaluate_combined_signal(s['s1'], s['s2'], "AND")
    for m in ['baseline', 'dca', 'partial_tp']:
        res = run_advanced_backtest(df, l_sig, None, s['name'], m, s['lev'], s['sl'], s['tp'])
        print(f"{s['name']:<25} | {m:<10} | {res['Return']:>7.2f}% | {res['MDD']:>7.2f}% | {res['WinRate']:>7.2f}%")
    print("-" * 75)
