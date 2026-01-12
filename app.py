import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import vectorbt as vbt
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import data_loader
import backtest
import strategy
import time
from trading_engine import TradingEngine
from saved_strategies_manager import SavedStrategyManager

st.set_page_config(page_title="BTC Strategy Miner V3", layout="wide", page_icon="⛏️")

# --- Custom CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; transition: all 0.2s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .metric-card { background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .sidebar-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; color: #4CAF50; }
    .sidebar-section { padding: 1rem; border-radius: 10px; margin-bottom: 1rem; background: rgba(255, 255, 255, 0.05); }
    .hof-card { padding: 10px; border: 1px solid #444; border-radius: 8px; margin-bottom: 5px; background-color: #222; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Control Center
# ---------------------------------------------------------
with st.sidebar:
    st.title("⛏️ Control Center")
    
    # 1. Mining Controls
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 🤖 실험실 (Auto Miner)")
    
    if 'mining_active' not in st.session_state: st.session_state.mining_active = False
    
    c1, c2 = st.columns(2)
    if c1.button("🔥 Start", type="primary", use_container_width=True):
        st.session_state.mining_active = True
        st.rerun()
    if c2.button("🛑 Stop", use_container_width=True):
        st.session_state.mining_active = False
        st.rerun()
        
    @st.fragment(run_every=2)
    def render_mining_status_sidebar():
        if 'composer' not in st.session_state: return
        composer = st.session_state.composer
        count = st.session_state.get('strategies_tested_count', 0)
        st.caption(f"Scanned: {count:,}")
        if st.session_state.get('mining_active', False):
            st.warning("⛏️ Mining in progress...") 
            
            BATCH_SIZE = 500
            new_batch_df = composer.generate_and_test(n_strategies=BATCH_SIZE, train_ratio=0.7)
            st.session_state.strategies_tested_count = st.session_state.get('strategies_tested_count', 0) + BATCH_SIZE
            if not new_batch_df.empty:
                top_10 = composer.load_hall_of_fame()
                st.session_state.hof_data_cache = top_10
    
    render_mining_status_sidebar()
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Saved Strategies
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### 📚 Strategy Library")
    
    if 'saved_strategy_manager' not in st.session_state: st.session_state.saved_strategy_manager = SavedStrategyManager()
    if 'active_strategy_idx' not in st.session_state: st.session_state.active_strategy_idx = 0
    saved_strats = st.session_state.saved_strategy_manager.get_strategies()
    
    if saved_strats:
        # Improved labels: Rank + Description + Score
        options = []
        for i, s in enumerate(saved_strats):
            desc = s.get('Description', 'Untitled')[:22]
            score = s.get('Score', 0)
            options.append(f"{i+1}. {desc}.. (🏆{score:.1f})")

        sel_idx = st.selectbox("Select Strategy", options=range(len(options)), format_func=lambda x: options[x], index=st.session_state.active_strategy_idx, label_visibility="collapsed")
        if sel_idx != st.session_state.active_strategy_idx: st.session_state.active_strategy_idx = sel_idx; st.rerun()
        
        # Strategy Detail Summary
        curr = saved_strats[sel_idx]
        st.markdown(f"""
        <div style='font-size: 0.85rem; color: #ffffff; background: #262626; padding: 12px; border-radius: 8px; border: 1px solid #444; margin: 10px 0;'>
            <div style='margin-bottom: 5px;'>📈 <b>Monthly Return:</b> <span style='color: #00E676; font-weight: bold;'>{curr.get('Monthly_Return', 0)*100:.1f}%</span></div>
            <div style='margin-bottom: 5px;'>🛡️ <b>MDD:</b> <span style='color: #FF5252;'>{curr.get('MDD', 0)*100:.1f}%</span> | 💰 <b>PF:</b> <span style='color: #FFD700;'>{curr.get('Profit_Factor', 0):.2f}</span></div>
            <div style='font-size: 0.75rem; color: #999; margin-top: 8px; border-top: 1px solid #444; padding-top: 6px;'>🕒 Saved: {curr.get('Added_Date', 'Unknown')}</div>
        </div>
        """, unsafe_allow_html=True)

        c_del1, c_del2 = st.columns(2)
        if c_del1.button("🗑️ Delete Current", use_container_width=True): 
            st.session_state.saved_strategy_manager.delete_strategy(sel_idx)
            st.session_state.active_strategy_idx = 0
            st.rerun()
            
        if c_del2.button("⚠️ Delete All", use_container_width=True, help="저장된 모든 전략을 삭제합니다"):
            st.session_state.saved_strategy_manager.delete_all_strategies()
            st.session_state.active_strategy_idx = 0
            st.rerun()
    else: st.info("No saved strategies.")
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Data Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ Data")
    symbol = st.text_input("Symbol", "BTC/USDT")
    if st.button("🔄 Refresh Data"): st.cache_data.clear(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Page Logic
# ---------------------------------------------------------
st.title("BTC Strategy Miner V3 🚀")

@st.cache_data
def get_data(symbol, refresh=False):
    try:
        df = data_loader.load_data("btc_futures_data_5m.csv")
        if df is None or refresh:
            df = data_loader.fetch_binance_futures_data(symbol, timeframe="5m", since_years=1)
            data_loader.save_data(df, "btc_futures_data_5m.csv")
        return df
    except: return None

with st.spinner("Loading Data..."):
    df = get_data(symbol)
    if df is not None and not df.empty:
        st.session_state.df_global = df
        if 'composer' not in st.session_state: 
            st.session_state.composer = backtest.RandomStrategyComposer(df)
        elif not hasattr(st.session_state.composer, 'delete_strategy'):
            st.session_state.composer = backtest.RandomStrategyComposer(df)
    else:
        st.error("❌ 데이터를 불러올 수 없습니다. 바이낸스 API가 이 서버의 IP를 차단했거나 CSV 파일이 없습니다.")
        if df is not None and df.empty:
            st.warning("⚠️ 데이터프레임이 비어 있습니다. 잠시 후 다시 시도하거나 다른 심볼을 선택하세요.")
        st.stop()

            
        current_ts = time.time()
        if (current_ts % 300) < 10 and ('last_refresh' not in st.session_state or current_ts - st.session_state.last_refresh > 60):
             st.session_state.last_refresh = current_ts; st.cache_data.clear(); st.rerun()

if df is not None:
    # 1. Resolve Active Strategy (Global)
    active_strat = None
    if saved_strats and sel_idx >= 0 and sel_idx < len(saved_strats):
        active_strat = saved_strats[sel_idx]
        st.session_state.active_strategy = active_strat
    else: active_strat = None
    
    selected_rank = st.session_state.active_strategy_idx + 1 if active_strat else 0

    with st.spinner("Analyzing Market & Strategy..."):
        # 2. Prepare Data & Signals
        ind = strategy.get_expanded_indicators(df)
        repo = strategy.SignalRepository(ind)
        df_strat = df.copy()
        
        if active_strat:
            lt = active_strat.get('logic_tuple')
            if lt:
                 keys, _, _ = repo.get_signal_matrix()
                 s1 = keys[lt[0]]; s2 = keys[lt[1]]; op = 'AND' if lt[2]==0 else 'OR'
                 f = keys[lt[3]] if lt[3]!=-1 else None 
                 e = repo.evaluate_combined_signal(s1, s2, op)
                 if f: e = e & repo.signals[f]
                 s_e = repo.evaluate_combined_signal(*repo.get_symmetric_logic(s1, s2, op)[:3])
                 if f: s_e = s_e & repo.signals[f]
                 df_strat['long_signal'] = e; df_strat['short_signal'] = s_e
            else:
                 df_strat['long_signal'] = False; df_strat['short_signal'] = False
            
            # Get params from strat
            lev = active_strat.get('Leverage', 1)
            sl = active_strat.get('SL_Pct', 0.03)
            tp = active_strat.get('TP_Pct', 0.1)
            
            pf = backtest.run_dca_backtest(df_strat, df_strat['long_signal'], df_strat['short_signal'], leverage=lev, base_sl=sl, base_tp=tp, return_portfolio=True)
        else:
             df_strat = strategy.apply_liquidity_sweep_strategy(df.copy(), 0.005, 30)
             pf = backtest.run_dca_backtest(df_strat, df_strat['long_signal'], df_strat['short_signal'], leverage=5, base_sl=0.03, base_tp=0.1, return_portfolio=True)

        metrics = {
            'Return': pf.total_return() * 100,
            'MDD': pf.max_drawdown() * 100,
            'WinRate': pf.trades.win_rate().iloc[0],
            'Trades': pf.trades.count()
        }
        
    last_close = df['close'].iloc[-1]
    last_sma = df['close'].rolling(200).mean().iloc[-1]
    is_long = df_strat['long_signal'].iloc[-1]
    is_short = df_strat['short_signal'].iloc[-1]

    # TOP HEADER: Current Market Status
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns([1, 1, 1, 1.5])
        m1.metric("현재가", f"${last_close:,.2f}")
        m2.metric("SMA 200", f"${last_sma:,.2f}")
        
        # Signals as clear badges
        if is_long: m3.success("Signal: LONG 🟢")
        elif is_short: m3.error("Signal: SHORT 🔴")
        else: m3.info("Signal: WAIT 🛡️")
        
        # Strategy Info with better alignment
        if active_strat:
            m4.markdown(f"**활성 전략:** `{active_strat['Description'][:25]}..`")
            m4.caption(f"🚀 Lev: {active_strat.get('Leverage', 1)}x | 📈 Monthly: {active_strat.get('Monthly_Return',0)*100:.1f}%")
        else:
            m4.warning("⚠️ 전략을 선택해주세요")

    # ---------------------------------------------------------
    # UI TABS
    # ---------------------------------------------------------
    t_hof, t_perf, t_live = st.tabs(["🏆 명예의 전당", "📈 전략 성과", "🔴 실전/모의 매매"])
     
    with t_hof:
        c_title, c_sync, c_clear = st.columns([2.5, 1, 1])
        c_title.markdown("### 🏆 명예의 전당 (Top Strategies)")
        if c_sync.button("🔄 성과 반영", use_container_width=True, help="기존 전략들에 스위칭 로직 및 최신 지표를 재계산하여 적용합니다."):
             with st.spinner("데이터 동기화 중..."):
                  st.session_state.composer.recalculate_hof_metrics()
                  st.success("✅ 업데이트 완료!")
                  time.sleep(1); st.rerun()
        
        if c_clear.button("🗑️ 전체 삭제", use_container_width=True, help="명예의 전당의 모든 전략을 비웁니다."):
             st.session_state.composer.delete_all_strategies()
             st.success("✅ 명예의 전당 초기화 완료"); time.sleep(1); st.rerun()

        st.caption("실험실에서 발견된 최고의 전략들입니다. '저장' 버튼을 눌러 내 라이브러리에 추가하세요.")
        
        @st.fragment(run_every=5)
        def render_hof_list():
             if 'composer' not in st.session_state: return
             composer = st.session_state.composer
             top_10 = composer.load_hall_of_fame()
             
             if not top_10:
                 st.info("아직 발견된 전략이 없습니다. 사이드바에서 실험실을 시작하세요!")
                 return
             
             # Header
             h1, h2, h3, h4, h5 = st.columns([0.5, 3, 1, 1, 1])
             h1.markdown("**Rank**")
             h2.markdown("**Strategy**")
             h3.markdown("**Monthly**")
             h4.markdown("**Score**")
             h5.markdown("**Action**")
             st.divider()
             
             today_str = datetime.now().strftime("%Y-%m-%d")
             for i, strat in enumerate(top_10):
                 is_new = strat.get('Added_Date', '').startswith(today_str)
                 new_badge = " <span style='background-color: #FF1744; color: white; padding: 1px 6px; border-radius: 10px; font-size: 0.65rem; font-weight: 800; vertical-align: middle; margin-left: 5px; box-shadow: 0 2px 4px rgba(255,23,68,0.3);'>NEW</span>" if is_new else ""
                 
                 with st.container():
                     c1, c2, c3, c4, c5 = st.columns([0.5, 3, 1, 1, 1])
                     c1.markdown(f"<h3 style='margin:0; color: #FFD700;'>#{i+1}</h3>", unsafe_allow_html=True)
                     c2.markdown(f"**{strat['Description']}**{new_badge}", unsafe_allow_html=True)
                     c2.caption(f"🕒 발견: {strat.get('Added_Date', 'Unknown')}")
                     
                     m_ret = strat['Monthly_Return']*100
                     c3.markdown(f"📈 {m_ret:.1f}%")
                     c4.markdown(f"🏆 {strat['Score']:.2f}")
                     
                     # Action Buttons Row
                     btn_c1, btn_c2, btn_c3 = c5.columns([1.2, 1.2, 0.8])
                     
                     # Check if already saved
                     saved_descriptions = [s.get('Description') for s in saved_strats]
                     is_already_saved = strat.get('Description') in saved_descriptions
                     
                     if btn_c1.button("💾 저장", key=f"save_btn_{i}", help="내 라이브러리에 저장", disabled=is_already_saved):
                         if st.session_state.saved_strategy_manager.save_strategy(strat):
                             st.toast(f"✅ 전략 #{i+1} 저장 완료!")
                             time.sleep(0.5) 
                             st.rerun() 
                     
                     mining_active = bool(st.session_state.get('mining_active', False))
                     is_unoptimizable = bool(strat.get('unoptimizable', False))
                     
                     enhance_label = "❌ " if is_unoptimizable else "🔥 "
                     help_msg = "더 이상 개선불가" if is_unoptimizable else "강화"
                     
                     if btn_c2.button(enhance_label, key=f"enhance_btn_{i}", help=help_msg, disabled=mining_active or is_unoptimizable, type="secondary"):
                         with st.spinner(f"강화 중..."):
                             success, msg = st.session_state.composer.optimize_strategy_params(i)
                             if success: st.success("🔥 성공!"); time.sleep(1); st.rerun()
                             else: st.info(f"알림: {msg}"); time.sleep(1); st.rerun()

                     if btn_c3.button("🗑️", key=f"hof_del_{i}", help="이 전략을 명예의 전당에서 삭제"):
                          st.session_state.composer.delete_strategy(i)
                          st.rerun()

                     # Quality Metrics Row
                     q1, q2, q3, q4, q5, q6 = st.columns([0.5, 1.2, 1.2, 1.2, 1.2, 1.2])
                     
                     pf_val = strat.get('Profit_Factor', 0)
                     sh = strat.get('Sharpe', 0)
                     so = strat.get('Sortino', 0)
                     mc = strat.get('Max_Cons_Loss', 0)
                     lev = strat.get('Leverage', 1)
                     
                     q2.caption(f"🚀 Lev: **{lev}x**")
                     q3.caption(f"💰 PF: **{pf_val:.2f}**")
                     q4.caption(f"⚡ Sharpe: **{sh:.2f}**")
                     q5.caption(f"🛡️ Sortino: **{so:.2f}**")
                     q6.caption(f"🩸 Max Loss: **{mc}**")
                 
                 st.markdown("<hr style='margin: 5px 0; opacity: 0.1;'>", unsafe_allow_html=True)
                 
        render_hof_list()




    with t_perf:
        st.subheader("📈 전략 성과 리포트")
        if active_strat:
            st.info(f"**현재 분석 전략:** {active_strat.get('Description')} (Lev: {active_strat.get('Leverage')}x)")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총 수익률", f"{metrics['Return']:.2f}%")
        m2.metric("승률 (Win Rate)", f"{metrics['WinRate']*100:.1f}%")
        m3.metric("최대 낙폭 (MDD)", f"{metrics['MDD']:.2f}%")
        m4.metric("총 거래 횟수", f"{metrics['Trades']}")
        
        if hasattr(pf, 'adv_metrics'):
            st.divider()
            a1, a2, a3, a4 = st.columns(4)
            am = pf.adv_metrics
            a1.metric("Profit Factor", f"{am['Profit_Factor']:.2f}")
            a2.metric("Sharpe Ratio", f"{am['Sharpe']:.2f}")
            a3.metric("Sortino Ratio", f"{am['Sortino']:.2f}")
            a4.metric("Max Cons. Loss", f"{am['Max_Cons_Loss']}")
        
        st.divider()
        
        pf_value = pf.value()
        start_price = df['close'].iloc[0]
        initial_equity = pf_value.iloc[0] 
        bnh_value = (df['close'] / start_price) * initial_equity
        
        fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08,
                              subplot_titles=("Equity Curve vs Buy&Hold", "Trades on BTC Price"))
        
        # Row 1: Equity Curve
        fig_p.add_trace(go.Scatter(x=pf_value.index, y=pf_value, line=dict(color='#00E676', width=2), name='My Strategy'), row=1, col=1)
        fig_p.add_trace(go.Scatter(x=bnh_value.index, y=bnh_value, line=dict(color='gray', dash='dot'), name='Buy & Hold'), row=1, col=1)

        # Row 2: Price Candlesticks & Trade Markers
        fig_p.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        ), row=2, col=1)
        
        trades_df = pf.trades.records_readable
        if not trades_df.empty:
            entries = trades_df[['Entry Index', 'Entry Price', 'Direction']]
            exits = trades_df[['Exit Index', 'Exit Price', 'Direction']]
            
            # Long Entries
            l_pentries = entries[entries['Direction'] == 'Long']
            if not l_pentries.empty:
                fig_p.add_trace(go.Scatter(
                    x=l_pentries['Entry Index'], y=l_pentries['Entry Price'],
                    mode='markers', marker=dict(symbol='triangle-up', color='#00E676', size=10),
                    name='Buy'
                ), row=2, col=1)
            
            # Short Entries
            s_pentries = entries[entries['Direction'] == 'Short']
            if not s_pentries.empty:
                fig_p.add_trace(go.Scatter(
                    x=s_pentries['Entry Index'], y=s_pentries['Entry Price'],
                    mode='markers', marker=dict(symbol='triangle-down', color='#FF1744', size=10),
                    name='Short Sell'
                ), row=2, col=1)
                
            # Exits
            if not exits.empty:
                fig_p.add_trace(go.Scatter(
                    x=exits['Exit Index'], y=exits['Exit Price'],
                    mode='markers', marker=dict(symbol='x', color='#FDD835', size=8),
                    name='Close'
                ), row=2, col=1)
        
        fig_p.update_layout(height=800, template="plotly_dark", hovermode="x unified")
        fig_p.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig_p, use_container_width=True)
        
        st.subheader("📝 상세 매매 기록")
        if not trades_df.empty:
            # 1. Prepare Data: Calculate Balance & Direction symbols
            t_df = trades_df.copy()
            
            # Robust Column Mapping
            col_lowers = {c.lower(): c for c in t_df.columns}
            if 'pnl' in col_lowers and 'PnL' not in t_df.columns:
                t_df['PnL'] = t_df[col_lowers['pnl']]
            if 'ret' in col_lowers and 'Return' not in t_df.columns:
                t_df['Return'] = t_df[col_lowers['ret']]
            
            t_df = t_df.sort_values('Exit Index')
            t_df['Cumulative PnL'] = t_df['PnL'].cumsum()
            t_df['Balance'] = 10000.0 + t_df['Cumulative PnL']
            
            # Convert decimal return to percentage (0.04 -> 4.0)
            t_df['Return_Pct'] = t_df['Return'] * 100.0
            
            # Format Direction for better visuals
            t_df['Side'] = t_df['Direction'].apply(lambda x: "🟢 Long" if str(x).lower() == 'long' else "🔴 Short")
            
            # 2. Display with Column Config
            display_df = t_df.sort_values('Entry Index', ascending=False)
            
            st.dataframe(
                display_df[['Side', 'Entry Index', 'Exit Index', 'Entry Price', 'Exit Price', 'PnL', 'Return_Pct', 'Balance']],
                column_config={
                    "Side": st.column_config.TextColumn("Trade Side"),
                    "Entry Index": st.column_config.DatetimeColumn("Entry Time", format="MM/DD HH:mm"),
                    "Exit Index": st.column_config.DatetimeColumn("Exit Time", format="MM/DD HH:mm"),
                    "Entry Price": st.column_config.NumberColumn("Entry ($)", format="$%.pd"), # Actually standard number is better for BTC
                    "Exit Price": st.column_config.NumberColumn("Exit ($)", format="$%.pd"),
                    "PnL": st.column_config.NumberColumn("PnL ($)", format="$%.2f"),
                    "Return_Pct": st.column_config.NumberColumn("Return (%)", format="%.2f%%"),
                    "Balance": st.column_config.NumberColumn("Account Balance", format="$%.2f")
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("거래 기록이 없습니다.")

    with t_live:
        st.subheader("🔴 실전/모의 매매 봇")
        
        # MARKET CONTEXT (Recent 1 Week)
        with st.expander("📊 최근 시장 지표 (1주일)", expanded=True):
            sub_df = df.tail(2016)
            min_ts = sub_df.index[0]
            fig_l = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            
            # 1. Price & SMA
            fig_l.add_trace(go.Candlestick(x=sub_df.index, open=sub_df['open'], high=sub_df['high'], low=sub_df['low'], close=sub_df['close'], name='BTC'), row=1, col=1)
            sma_l = df['close'].rolling(200).mean().reindex(sub_df.index)
            fig_l.add_trace(go.Scatter(x=sub_df.index, y=sma_l, line=dict(color='rgba(255,255,0,0.4)', width=1), name='SMA 200'), row=1, col=1)
            
            # 2. Actual Trades (Executed) within 1 week
            if hasattr(pf, 'trades') and not pf.trades.records_readable.empty:
                t_recs = pf.trades.records_readable
                
                # Filter Entry markers (Long/Short) strictly within view
                l_ent = t_recs[(t_recs['Direction'] == 'Long') & (t_recs['Entry Index'] >= min_ts)]
                if not l_ent.empty:
                    fig_l.add_trace(go.Scatter(x=l_ent['Entry Index'], y=l_ent['Entry Price'], mode='markers', 
                                             marker=dict(symbol='diamond', color='#00E676', size=12, line=dict(width=1, color='white')), name='L-Entry'), row=1, col=1)
                
                s_ent = t_recs[(t_recs['Direction'] == 'Short') & (t_recs['Entry Index'] >= min_ts)]
                if not s_ent.empty:
                    fig_l.add_trace(go.Scatter(x=s_ent['Entry Index'], y=s_ent['Entry Price'], mode='markers', 
                                             marker=dict(symbol='diamond', color='#FF1744', size=12, line=dict(width=1, color='white')), name='S-Entry'), row=1, col=1)
                
                # Filter Exit markers strictly within view
                v_exits = t_recs[t_recs['Exit Index'] >= min_ts]
                if not v_exits.empty:
                    fig_l.add_trace(go.Scatter(x=v_exits['Exit Index'], y=v_exits['Exit Price'], mode='markers', 
                                             marker=dict(symbol='x', color='#FDD835', size=10), name='Exit'), row=1, col=1)

            fig_l.update_layout(height=450, template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
            fig_l.update_xaxes(range=[min_ts, sub_df.index[-1]], rangeslider_visible=False)
            st.plotly_chart(fig_l, use_container_width=True)

        if active_strat:
             st.info(f"**적용 전략:** {active_strat.get('Description')} (Lev: {active_strat.get('Leverage')}x)")
        else:
             st.warning("⚠️ 기본 전략(Liquidity Sweep)이 적용 중입니다.")

        st.markdown("⚠️ **DCA 전략 적용됨:** 진입 후 -2% 하락 시 2차 매수(배수)가 자동으로 실행됩니다.")
        
        # 2. Trading Mode & API Settings
        t_mode = st.radio("매매 모드 선택", ["📝 Paper Trading (모의투자)", "🔥 Live Trading (실전매매)"], horizontal=True)
        is_live = "Live" in t_mode
        
        api_key, api_secret = None, None
        if is_live:
            st.markdown("#### 🔑 Binance API 설정")
            c_k1, c_k2 = st.columns(2)
            api_key = c_k1.text_input("Binance API Key", type="password", key="live_api_key", help="바이낸스 선물 API 키를 입력하세요.")
            api_secret = c_k2.text_input("Binance Secret Key", type="password", key="live_api_secret", help="바이낸스 선물 Secret 키를 입력하세요.")
            st.divider()
        
        col_inp1, col_inp2 = st.columns(2)
        alloc_amt = col_inp1.number_input("1회 진입 금액 (USDT)", min_value=10.0, value=50.0, help="DCA 1차 진입 시 사용할 원금입니다.")
        
        if 'live_trading_active' not in st.session_state: st.session_state.live_trading_active = False
        
        if st.button("🚀 봇 시작 / 재설정", type="primary"):
            mode_str = 'Live' if is_live else 'Paper'
            
            if is_live and (not api_key or not api_secret):
                st.error("❌ 실전 매매는 API Key가 필요합니다.")
            else:
                try:
                    eng = TradingEngine(api_key, api_secret, mode=mode_str)
                    st.session_state.live_engine = eng
                    st.session_state.live_trading_active = True
                    st.toast(f"✅ {mode_str} Trading Started!")
                    st.rerun()
                except Exception as e: st.error(f"초기화 실패: {e}")
                
        if st.session_state.live_trading_active:
             engine = st.session_state.live_engine
             if 'live_engine' in st.session_state and engine:
                 st.success(f"🟢 {engine.mode} Bot Running...")
                 
                 c_stop, c_logs = st.columns([1, 4])
                 if c_stop.button("🛑 봇 정지"):
                     st.session_state.live_trading_active = False; st.session_state.live_engine = None; st.rerun()
                 
                 lev = active_strat.get('Leverage', 1) if active_strat else 1
                 sl_val = active_strat.get('SL_Pct', 0.03) if active_strat else 0.03
                 tp_val = active_strat.get('TP_Pct', 0.1) if active_strat else 0.1
                 
                 engine.sync_and_execute(is_long, is_short, last_close, alloc_amt, leverage=lev, sl=sl_val, tp=tp_val)
                 
                 st.code("\n".join(engine.logs[-15:]))
             else:
                 st.error("Engine lost. Restart.")
