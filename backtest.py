import vectorbt as vbt
import numpy as np
import pandas as pd
import strategy
import random
import os
import json
from datetime import datetime
from numba import njit
import sys

# --- Optimized Core Logic (Numba JIT) with SWITCHING ---
@njit
def _fast_dca_core_v2(closes, highs, lows, long_signals, short_signals, leverage, base_sl=0.03, base_tp=0.1):
    n = len(closes)
    
    # Output arrays
    # t_entry_idx, t_exit_idx, t_entry_price, t_exit_price, t_pnl, t_ret, t_direction (1=Long, -1=Short)
    out_trades = np.zeros((n, 7)) 
    trade_count = 0
    
    balance = 10000.0
    position_size = 0.0 # Positive for Long, Negative for Short? No, use explicit direction.
    position_amt = 0.0 # Absolute amount in USDT
    
    # Position State
    # 0: Flat, 1: Long, -1: Short
    pos_dir = 0 
    avg_entry_price = 0.0
    dca_filled = False
    entry_fee = 0.0004
    
    equity_curve = np.zeros(n)
    
    # Temp vars for trade recording
    current_entry_idx = -1
    
    for i in range(n):
        curr_close = closes[i]
        curr_high = highs[i]
        curr_low = lows[i]
        
        # Sync Balance
        if i > 0: equity_curve[i] = equity_curve[i-1]
        else: equity_curve[i] = balance
        
        # --- 1. Manage Existing Position ---
        if pos_dir != 0:
            
            # A. Check DCA
            if pos_dir == 1: # Long
                dca_price = avg_entry_price * 0.98
                if not dca_filled and curr_low <= dca_price:
                    dca_usdt = position_amt # 1:1 add
                    qty_old = position_amt / avg_entry_price
                    qty_new = dca_usdt / dca_price
                    position_amt += dca_usdt
                    avg_entry_price = position_amt / (qty_old + qty_new)
                    dca_filled = True
            
            elif pos_dir == -1: # Short
                dca_price = avg_entry_price * 1.02 # Add if price RISES 2%
                if not dca_filled and curr_high >= dca_price:
                    dca_usdt = position_amt
                    qty_old = position_amt / avg_entry_price
                    qty_new = dca_usdt / dca_price
                    position_amt += dca_usdt
                    # For short, avg price is total_value / total_qty too
                    avg_entry_price = position_amt / (qty_old + qty_new)
                    dca_filled = True

            # B. Check Exits (SL/TP)
            exit_triggered = False
            actual_exit_price = 0.0
            
            if pos_dir == 1: # Long Exits
                exit_sl = avg_entry_price * (1 - base_sl) # e.g. 0.95
                exit_tp = avg_entry_price * (1 + base_tp) # e.g. 1.10
                
                if curr_low <= exit_sl:
                    actual_exit_price = exit_sl
                    if curr_high < exit_sl: actual_exit_price = curr_close # Gap
                    exit_triggered = True
                elif curr_high >= exit_tp:
                    actual_exit_price = exit_tp
                    if curr_low > exit_tp: actual_exit_price = curr_close
                    exit_triggered = True
                    
            elif pos_dir == -1: # Short Exits
                exit_sl = avg_entry_price * (1 + base_sl) # SL for short is Higher
                exit_tp = avg_entry_price * (1 - base_tp) # TP for short is Lower
                
                if curr_high >= exit_sl:
                    actual_exit_price = exit_sl
                    if curr_low > exit_sl: actual_exit_price = curr_close
                    exit_triggered = True
                elif curr_low <= exit_tp:
                    actual_exit_price = exit_tp
                    if curr_high < exit_tp: actual_exit_price = curr_close
                    exit_triggered = True

            # C. Check Switching (Signal Reversal)
            # If Long and Short Signal triggers -> Switch
            # Only if not already exiting by SL/TP
            switch_signal = False
            if not exit_triggered:
                if pos_dir == 1 and short_signals[i]:
                    switch_signal = True
                    actual_exit_price = curr_close
                    exit_triggered = True
                elif pos_dir == -1 and long_signals[i]:
                    switch_signal = True
                    actual_exit_price = curr_close
                    exit_triggered = True

            # Execute Exit
            if exit_triggered:
                # Calculate PnL
                raw_ret = 0.0
                if pos_dir == 1:
                    raw_ret = (actual_exit_price - avg_entry_price) / avg_entry_price
                else:
                    raw_ret = (avg_entry_price - actual_exit_price) / avg_entry_price
                
                pnl = position_amt * raw_ret
                balance += pnl - (position_amt * entry_fee)
                
                # Record
                out_trades[trade_count, 0] = current_entry_idx
                out_trades[trade_count, 1] = i
                out_trades[trade_count, 2] = avg_entry_price
                out_trades[trade_count, 3] = actual_exit_price
                out_trades[trade_count, 4] = pnl
                out_trades[trade_count, 5] = raw_ret
                out_trades[trade_count, 6] = pos_dir
                trade_count += 1
                
                equity_curve[i] = balance
                
                # Reset
                pos_dir = 0
                position_amt = 0.0
                dca_filled = False
                
                # If Switch (Reverse), we need to Open New Position immediately in SAME BAR?
                # Simply setting pos_dir=0 allows the "Entry Logic" below to pick it up if signal persists.
                # However, we used 'curr_close' for exit. Entry will also use 'curr_close'.
                
        # --- 2. Check Entry ---
        if pos_dir == 0:
            # Check Long
            if long_signals[i]:
                pos_dir = 1
                position_amt = balance * leverage * 0.5
                avg_entry_price = curr_close
                dca_filled = False
                balance -= position_amt * entry_fee
                current_entry_idx = i
                equity_curve[i] = balance
             
            # Check Short (Else If to avoid double entry)
            elif short_signals[i]:
                pos_dir = -1
                position_amt = balance * leverage * 0.5
                avg_entry_price = curr_close
                dca_filled = False
                balance -= position_amt * entry_fee
                current_entry_idx = i
                equity_curve[i] = balance
            
    return equity_curve, out_trades[:trade_count]

class StrategyMiningEngine:
    def __init__(self, df):
        self.df = df
        
class MockTrades:
    def __init__(self, records_df, wr, count):
        self.records_readable = records_df
        self._wr = wr
        self._count = count
    def win_rate(self): return pd.Series([self._wr]) 
    def count(self): return self._count

class MockPortfolio:
    def __init__(self, equity_series, trades_df, mdd, ret, wr, total_trades):
        self._equity = equity_series
        self.trades = MockTrades(trades_df, wr, total_trades)
        self._mdd = mdd
        self._ret = ret
    def total_return(self): return self._ret
    def max_drawdown(self): return self._mdd
    def value(self): return self._equity

def run_dca_backtest(df, long_signals, short_signals, leverage=1, base_sl=0.03, base_tp=0.1, return_portfolio=False):
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    long_arr = long_signals.values.astype(np.bool_)
    
    # FIX: Handle Short Signals
    if short_signals is not None:
        short_arr = short_signals.values.astype(np.bool_)
    else:
        short_arr = np.zeros_like(long_arr, dtype=np.bool_)
    
    # Run Fast Core with Switching
    equity_curve, trades_arr = _fast_dca_core_v2(closes, highs, lows, long_arr, short_arr, float(leverage), base_sl, base_tp)
    
    if len(equity_curve) == 0:
        return 0.0, 0.0, 0.0, 0, {} if not return_portfolio else MockPortfolio(pd.Series(), pd.DataFrame(), 0.0, 0.0, 0.0, 0)

    balance = equity_curve[-1]

    ret_pct = (balance - 10000.0)/10000.0
    
    peak = np.maximum.accumulate(equity_curve)
    div = np.where(peak == 0, 1, peak)
    dd = (equity_curve - peak) / div
    mdd = np.min(dd)
    
    cnt = len(trades_arr)
    wins = np.sum(trades_arr[:, 5] > 0)
    wr = (wins/cnt) if cnt>0 else 0.0
    
    gross_profit = np.sum(trades_arr[:, 4][trades_arr[:, 4] > 0])
    gross_loss = abs(np.sum(trades_arr[:, 4][trades_arr[:, 4] <= 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    
    max_cons_loss = 0
    curr_cons_loss = 0
    for r in trades_arr[:, 5]: 
        if r < 0:
            curr_cons_loss += 1
            max_cons_loss = max(max_cons_loss, curr_cons_loss)
        else:
            curr_cons_loss = 0
            
    equity_s = pd.Series(equity_curve)
    returns = equity_s.pct_change().dropna()
    sharpe = 0.0
    sortino = 0.0
    if len(returns) > 10:
        mean_ret = returns.mean()
        std_ret = returns.std()
        if std_ret > 0: sharpe = (mean_ret / std_ret) * np.sqrt(288 * 365)
        downside = returns[returns < 0]
        down_std = downside.std()
        if down_std > 0: sortino = (mean_ret / down_std) * np.sqrt(288 * 365)
        
    adv_metrics = {
        'Profit_Factor': profit_factor,
        'Max_Cons_Loss': max_cons_loss,
        'Sharpe': sharpe,
        'Sortino': sortino
    }
    
    if return_portfolio:
        t_list = []
        times = df.index
        for i in range(cnt):
            e_idx = int(trades_arr[i, 0])
            x_idx = int(trades_arr[i, 1])
            direction = 'Long' if trades_arr[i, 6] == 1 else 'Short'
            t_list.append({
                'Direction': direction,
                'Entry Index': times[e_idx],
                'Exit Index': times[x_idx],
                'Entry Price': trades_arr[i, 2],
                'Exit Price': trades_arr[i, 3],
                'PnL': trades_arr[i, 4],
                'Return': trades_arr[i, 5]
            })
        trades_df = pd.DataFrame(t_list)
        if trades_df.empty:
             trades_df = pd.DataFrame(columns=['Direction', 'Entry Index', 'Exit Index', 'Entry Price', 'Exit Price', 'PnL', 'Return'])
             
        pf = MockPortfolio(pd.Series(equity_curve, index=df.index), trades_df, abs(mdd), ret_pct, wr, cnt)
        pf.adv_metrics = adv_metrics
        return pf

    return ret_pct, abs(mdd), wr, cnt, adv_metrics

def evaluate_strat(df, repo, keys, sig_mat, lt, leverage, sl=0.03, tp=0.1):
    s1, s2, op, s3 = lt
    n_feats = len(keys)
    k_map = {k: i for i, k in enumerate(keys)}
    
    # Long
    sig1, sig2 = sig_mat[:, s1], sig_mat[:, s2]
    e_long = sig1 & sig2 if op==0 else sig1 | sig2
    if s3 != -1: e_long = e_long & sig_mat[:, s3]
    
    # Short
    sym_s1, sym_s2, sym_op = repo.get_symmetric_logic(keys[s1], keys[s2], 'AND' if op==0 else 'OR')
    try:
        idx_sym1 = k_map[sym_s1]
        idx_sym2 = k_map[sym_s2]
        sig_sh1, sig_sh2 = sig_mat[:, idx_sym1], sig_mat[:, idx_sym2]
        e_short = sig_sh1 & sig_sh2 if sym_op=='AND' else sig_sh1 | sig_sh2
        if s3 != -1: e_short = e_short & sig_mat[:, s3]
    except:
        e_short = np.zeros_like(e_long)
        
    return run_dca_backtest(df, pd.Series(e_long, index=df.index), pd.Series(e_short, index=df.index), leverage, base_sl=sl, base_tp=tp)

class RandomStrategyComposer:
    def __init__(self, df):
        self.df = df
        self.hof_file = "hall_of_fame_v3.json"
        self.indicators = strategy.get_expanded_indicators(df)
        self.repo = strategy.SignalRepository(self.indicators)
        
    def recalculate_hof_metrics(self):
        hof = self.load_hall_of_fame()
        if not hof: return
        
        updated_hof = []
        keys, sig_mat, _ = self.repo.get_signal_matrix()
        k_map = {k: i for i, k in enumerate(keys)}
        print(f"Recalculating metrics for {len(hof)} strategies with Switching support...")
        for strat in hof:
            try:
                lt = strat.get('logic_tuple')
                if not lt: 
                    updated_hof.append(strat)
                    continue
                
                s1, s2, op, s3 = lt
                # Long
                sig1, sig2 = sig_mat[:, s1], sig_mat[:, s2]
                e_long = sig1 & sig2 if op==0 else sig1 | sig2
                if s3 != -1: e_long = e_long & sig_mat[:, s3]
                
                # Short (Symmetric)
                sym_s1, sym_s2, sym_op = self.repo.get_symmetric_logic(keys[s1], keys[s2], 'AND' if op==0 else 'OR')
                idx_sym1 = k_map[sym_s1]
                idx_sym2 = k_map[sym_s2]
                sig_sh1, sig_sh2 = sig_mat[:, idx_sym1], sig_mat[:, idx_sym2]
                e_short = sig_sh1 & sig_sh2 if sym_op=='AND' else sig_sh1 | sig_sh2
                # Note: Apply same filter to short or omit? Usually filter is market regime, so same.
                if s3 != -1: e_short = e_short & sig_mat[:, s3]

                r, m, w, t, adv = evaluate_strat(self.df, self.repo, keys, sig_mat, lt, strat.get('Leverage', 1), strat.get('SL_Pct', 0.03), strat.get('TP_Pct', 0.1))
                
                strat['Sharpe'] = adv['Sharpe']
                strat['Sortino'] = adv['Sortino']
                strat['Profit_Factor'] = adv['Profit_Factor']
                strat['Max_Cons_Loss'] = adv['Max_Cons_Loss']
                strat['Return'] = r
                strat['MDD'] = m
                strat['Win Rate'] = w
                strat['Trades'] = t
                delta_days = (self.df.index[-1] - self.df.index[0]).total_seconds() / 86400
                if delta_days < 0.1: delta_days = 0.1
                strat['Monthly_Return'] = ((1+r)**(30/delta_days)) - 1
                strat['Score'] = (r / max(m, 0.05)) * np.log10(max(t, 10))
                
                if 'Added_Date' not in strat:
                    strat['Added_Date'] = "2025-01-01 00:00"
                
                updated_hof.append(strat)
            except Exception as e:
                print(f"Error recalc strat {strat.get('Description')}: {e}")
                updated_hof.append(strat)
        
        self.save_hall_of_fame(pd.DataFrame(updated_hof), mode='overwrite')

    def generate_and_test(self, n_strategies=500, train_ratio=0.7):
        try:
            split_idx = int(len(self.df) * train_ratio)
            train_df = self.df.iloc[:split_idx]
            test_df = self.df.iloc[split_idx:]
            
            keys, sig_mat, _ = self.repo.get_signal_matrix()
            k_map = {k: i for i, k in enumerate(keys)}
            n_feats = len(keys)
            train_sig_mat = sig_mat[:split_idx]
            test_sig_mat = sig_mat[split_idx:]
            
            hof = self.load_hall_of_fame()
            n_genetic = int(n_strategies * 0.3) if len(hof) >= 2 else 0
            n_random = n_strategies - n_genetic
            
            candidates_pre = []
            
            if n_genetic > 0:
                parents = []
                for h in hof:
                    lt = h.get('logic_tuple')
                    if lt:
                        tp_val = h.get('TP_Pct')
                        parents.append({
                            'lt': lt,
                            'lev': h.get('Leverage', 1),
                            'sl': h.get('SL_Pct', 0.01),
                            'tp': float(tp_val) if tp_val is not None else np.nan
                        })
                
                if len(parents) >= 2:
                    for _ in range(n_genetic):
                        p1, p2 = random.choice(parents), random.choice(parents)
                        lt1, lt2 = p1['lt'], p2['lt']
                        
                        c_s1 = lt1[0] if random.random() > 0.5 else lt2[0]
                        c_s2 = lt1[1] if random.random() > 0.5 else lt2[1]
                        c_op = lt1[2] if random.random() > 0.5 else lt2[2]
                        c_s3 = lt1[3] if random.random() > 0.5 else lt2[3]
                        
                        c_lev = p1['lev'] if random.random() > 0.5 else p2['lev']
                        
                        if random.random() < 0.25:
                            mut = random.choice(['logic', 'param', 'filter'])
                            if mut == 'logic': c_s1 = random.randint(0, n_feats-1)
                            elif mut == 'param': c_lev = random.choice([1, 2, 3, 5, 10])
                            elif mut == 'filter': c_s3 = random.randint(0, n_feats-1) if random.random() > 0.5 else -1
                        
                        candidates_pre.append({
                            'logic_tuple': (c_s1, c_s2, c_op, c_s3),
                            'logic': (keys[c_s1], keys[c_s2], 'AND' if c_op==0 else 'OR', keys[c_s3] if c_s3!=-1 else None),
                            'Leverage': c_lev
                        })

            for _ in range(n_random):
                s1 = random.randint(0, n_feats-1)
                s2 = random.randint(0, n_feats-1)
                op = random.randint(0, 1) # 0: AND, 1: OR
                s3 = random.randint(0, n_feats-1) if random.random() > 0.7 else -1
                lev = random.choice([1, 2, 3, 5, 10])
                
                candidates_pre.append({
                    'logic_tuple': (s1, s2, op, s3),
                    'logic': (keys[s1], keys[s2], 'AND' if op==0 else 'OR', keys[s3] if s3!=-1 else None),
                    'Leverage': lev
                })
                
            # Vectorized Evaluation with Auto Switching
            rets = []
            mdds = []
            scores = []
            
            for i, cand in enumerate(candidates_pre):
                 s1, s2, op, s3 = cand['logic_tuple']
                 
                 # Long Logic
                 sig1 = train_sig_mat[:, s1]
                 sig2 = train_sig_mat[:, s2]
                 entry_long = sig1 & sig2 if op==0 else sig1 | sig2
                 if s3 != -1: entry_long = entry_long & train_sig_mat[:, s3]
                 
                 # Short Logic (Symmetric)
                 sym_s1, sym_s2, sym_op = self.repo.get_symmetric_logic(keys[s1], keys[s2], 'AND' if op==0 else 'OR')
                 try:
                     idx_sym1 = k_map[sym_s1]
                     idx_sym2 = k_map[sym_s2]
                     sig_sh1 = train_sig_mat[:, idx_sym1]
                     sig_sh2 = train_sig_mat[:, idx_sym2]
                     entry_short = sig_sh1 & sig_sh2 if sym_op=='AND' else sig_sh1 | sig_sh2
                     # Note: We omit filter for short for now as filter symmetry is seemingly undefined
                 except:
                     entry_short = np.zeros_like(entry_long)

                 long_series = pd.Series(entry_long, index=train_df.index)
                 short_series = pd.Series(entry_short, index=train_df.index)
                 
                 r, m, w, t, adv = run_dca_backtest(train_df, long_series, short_series, cand['Leverage'])
                 
                 rets.append(r)
                 mdds.append(m)
                 scores.append(r / max(m, 0.05))

            candidates = []
            for i in range(len(candidates_pre)):
                 if rets[i] > 0.01 and mdds[i] < 0.2: 
                      cand = candidates_pre[i]
                      cand['train_return'] = rets[i]
                      cand['train_score'] = scores[i]
                      candidates.append(cand)
            
            # OOS
            results = []
            for cand in candidates:
                lt = cand['logic_tuple']
                s1, s2, op, s3 = lt
                
                # Long
                sig1 = test_sig_mat[:, s1]
                sig2 = test_sig_mat[:, s2]
                entry_long = sig1 & sig2 if op==0 else sig1 | sig2
                if s3 != -1: entry_long = entry_long & test_sig_mat[:, s3]
                
                # Short
                sym_s1, sym_s2, sym_op = self.repo.get_symmetric_logic(keys[s1], keys[s2], 'AND' if op==0 else 'OR')
                try:
                     idx_sym1 = k_map[sym_s1]
                     idx_sym2 = k_map[sym_s2]
                     sig_sh1 = test_sig_mat[:, idx_sym1]
                     sig_sh2 = test_sig_mat[:, idx_sym2]
                     entry_short = sig_sh1 & sig_sh2 if sym_op=='AND' else sig_sh1 | sig_sh2
                except:
                     entry_short = np.zeros_like(entry_long)
                
                long_series = pd.Series(entry_long, index=test_df.index)
                short_series = pd.Series(entry_short, index=test_df.index)
                
                oos_ret, oos_mdd, wr, tr, adv = run_dca_backtest(test_df, long_series, short_series, cand['Leverage'])
                
                if oos_ret > 0:
                     d_days = (test_df.index[-1] - test_df.index[0]).total_seconds() / 86400
                     if d_days < 0.1: d_days = 0.1
                     monthly = ((1+oos_ret)**(30/d_days)) - 1
                     score = (oos_ret / max(oos_mdd, 0.05)) * np.log10(max(tr, 10))
                     results.append({
                         'Description': f"{cand['logic'][0]} {cand['logic'][2]} {cand['logic'][1]}",
                         'Logic': cand['logic'], 'logic_tuple': lt,
                         'Leverage': cand['Leverage'],
                         'Return': oos_ret, 'Win Rate': wr, 'Trades': tr,
                         'MDD': oos_mdd, 'Score': score, 'Monthly_Return': monthly,
                         'Sharpe': adv['Sharpe'],
                         'Sortino': adv['Sortino'],
                         'Profit_Factor': adv['Profit_Factor'],
                         'Max_Cons_Loss': adv['Max_Cons_Loss'],
                         'Added_Date': datetime.now().strftime("%Y-%m-%d %H:%M")
                     })
            
            results = sorted(results, key=lambda x: x['Score'], reverse=True)[:20]
            if len(results) > 0: self.save_hall_of_fame(pd.DataFrame(results))
            
            return pd.DataFrame(results)

        except Exception as e:
            print(f"Mining Error: {e}")
            import traceback; traceback.print_exc()
            return pd.DataFrame()

    def load_hall_of_fame(self):
        if not os.path.exists(self.hof_file): return []
        try:
            with open(self.hof_file, 'r') as f: return json.load(f)
        except: return []

    def optimize_strategy_params(self, strat_idx):
        hof = self.load_hall_of_fame()
        if strat_idx < 0 or strat_idx >= len(hof): return False, "Invalid Index"
        
        strat = hof[strat_idx]
        lt = strat.get('logic_tuple')
        if not lt: return False, "No logic found"
        
        best_score = strat.get('Score', 0)
        best_params = {
            'Leverage': strat.get('Leverage', 1),
            'SL_Pct': strat.get('SL_Pct', 0.03),
            'TP_Pct': strat.get('TP_Pct', 0.1)
        }
        
        # Ranges to search
        leverages = [1, 2, 3, 5, 10, 15, 20]
        sl_range = [0.01, 0.02, 0.03, 0.04, 0.05]
        tp_range = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
        
        keys, sig_mat, _ = self.repo.get_signal_matrix()
        improved = False
        
        # Use simpler score for optimization internally if needed, but here we stay consistent
        delta_days = (self.df.index[-1] - self.df.index[0]).total_seconds() / 86400
        if delta_days < 0.1: delta_days = 0.1
        
        for lev in leverages:
            for sl in sl_range:
                for tp in tp_range:
                    r, m, w, t, adv = evaluate_strat(self.df, self.repo, keys, sig_mat, lt, lev, sl, tp)
                    score = (r / max(m, 0.05)) * np.log10(max(t, 10))
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'Leverage': lev, 'SL_Pct': sl, 'TP_Pct': tp}
                        improved = True
                        # Update strat object for immediate feedback
                        strat.update({
                            'Leverage': lev, 'SL_Pct': sl, 'TP_Pct': tp,
                            'Return': r, 'MDD': m, 'Win Rate': w, 'Trades': t,
                            'Score': score,
                            'Monthly_Return': ((1+r)**(30/delta_days)) - 1,
                            'Sharpe': adv['Sharpe'], 'Sortino': adv['Sortino'],
                            'Profit_Factor': adv['Profit_Factor'], 'Max_Cons_Loss': adv['Max_Cons_Loss'],
                            'Description': strat['Description'].split(" [")[0] + f" [Enh-L{lev}-S{sl}-T{tp}]"
                        })

        if improved:
            hof[strat_idx] = strat
            # Clear unoptimizable flag if it was set (though unlikely if we improved)
            if 'unoptimizable' in hof[strat_idx]: del hof[strat_idx]['unoptimizable']
            self.save_hall_of_fame(pd.DataFrame(hof), mode='overwrite')
            return True, "Success"
        
        # Mark as unoptimizable
        hof[strat_idx]['unoptimizable'] = True
        self.save_hall_of_fame(pd.DataFrame(hof), mode='overwrite')
        return False, "No improvement found"

    def delete_strategy(self, index):
        hof = self.load_hall_of_fame()
        if 0 <= index < len(hof):
            hof.pop(index)
            self.save_hall_of_fame(pd.DataFrame(hof), mode='overwrite')
            return True
        return False

    def delete_all_strategies(self):
        self.save_hall_of_fame(pd.DataFrame(), mode='overwrite')
        return True

    def save_hall_of_fame(self, new_df, mode='merge'):
        current = self.load_hall_of_fame()
        
        if mode == 'overwrite':
             combined = new_df
        else:
            combined = pd.DataFrame(current + new_df.to_dict('records'))
            combined = combined.drop_duplicates(subset=['Description'])
        
        # Cleanup NaNs which cause issues with Streamlit/Protobuf
        if 'unoptimizable' in combined.columns:
            combined['unoptimizable'] = combined['unoptimizable'].fillna(False).astype(bool)
        
        # Fill other numeric NaNs with 0/default to avoid protocols issues
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        combined[numeric_cols] = combined[numeric_cols].fillna(0)
        
        combined = combined.sort_values(by='Score', ascending=False).head(20)
        
        # Use simple list of dicts for saving, ensuring JSON compatibility
        records = combined.to_dict('records')
        # Final scrub of any remaining NaNs (Pandas to_dict sometimes leaves them)
        import math
        for r in records:
            for k, v in r.items():
                if isinstance(v, float) and math.isnan(v):
                    r[k] = 0
        
        with open(self.hof_file, 'w') as f:
             json.dump(records, f, indent=4)
