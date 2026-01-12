import pandas as pd
import numpy as np
import strategy
import backtest
import data_loader

def refresh_hof():
    # Load Data
    print("Loading Data...")
    df = data_loader.load_data()
    if df is None: return

    # Init Engine
    composer = backtest.RandomStrategyComposer(df)
    
    # Load Current HOF
    current_hof = composer.load_hall_of_fame()
    if not current_hof:
        print("HOF Empty.")
        return

    print(f"Refreshing {len(current_hof)} strategies with DCA Logic...")
    
    train_ratio = 0.7
    split_idx = int(len(df) * train_ratio)
    test_df = df.iloc[split_idx:]
    
    # Get Signals
    keys, sig_mat, _ = composer.repo.get_signal_matrix()
    test_sig_mat = sig_mat[split_idx:]
    
    updated_records = []
    
    for strat in current_hof:
        try:
            lt = strat['logic_tuple']
            s1, s2, op, f = lt[0], lt[1], lt[2], lt[3]
            
            # Reconstruct Signal
            sig1 = test_sig_mat[:, s1]
            sig2 = test_sig_mat[:, s2]
            entry = sig1 & sig2 if op == 0 else sig1 | sig2
            
            if f != -1:
                entry = entry & test_sig_mat[:, f]
                
            long_s = pd.Series(entry, index=test_df.index)
            
            # Run DCA
            lev = strat.get('Leverage', 1)
            
            r, mdd, wr, tr = backtest.run_dca_backtest(test_df, long_s, None, lev)
            
            # Calc Score
            test_days = (test_df.index[-1] - test_df.index[0]).total_seconds() / (24 * 3600)
            if test_days < 1: test_days = 1
            monthly_ret = ((1 + r) ** (30 / test_days)) - 1
            
            tm = np.log10(max(tr, 10))
            score = (r / max(mdd, 0.05)) * tm
            
            if mdd < 0.95 and r > -0.2:
                # Update
                strat['Return'] = float(r)
                strat['MDD'] = float(mdd)
                strat['Win Rate'] = float(wr)
                strat['Trades'] = int(tr)
                strat['Score'] = float(score)
                strat['Monthly_Return'] = float(monthly_ret)
                strat['SL_Pct'] = 0.05
                strat['TP_Pct'] = 0.10
                updated_records.append(strat)
            else:
                 print(f"Strategy {strat['Description']} failed validation with DCA. Dropping.")
                 
        except Exception as e:
            print(f"Error updating {strat.get('Description')}: {e}")
            import traceback; traceback.print_exc()
            
    # Save back
    if updated_records:
        df_new = pd.DataFrame(updated_records)
        df_new = df_new.sort_values(by='Score', ascending=False)
        print("Top 5 Updated:")
        print(df_new[['Description', 'Return', 'MDD', 'Score']].head())
        # Force Save
        composer.save_hall_of_fame(df_new)
        print("HOF Updated.")

if __name__ == "__main__":
    refresh_hof()
