import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    Client = None
    BinanceAPIException = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

# Setup basic logger for tenacity
logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, api_key=None, api_secret=None, mode='Paper'):
        self.mode = mode
        self.client = None
        self.active_trade = None # {side, entry_price, amount, dca_filled...}
        self.logs = []
        
        if mode == 'Live' and api_key and api_secret:
            try:
                self.client = Client(api_key, api_secret)
                self.log("✅ Binance Client Initialized")
            except Exception as e:
                self.log(f"❌ Failed to connect: {e}")
        else:
            self.mode = 'Paper' # Fallback
            self.log(f"📝 {mode if mode=='Paper' else 'Paper (Fallback)'} Trading Mode Initialized")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        self.logs.append(log_msg)
        print(log_msg) 
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]

    def _log_retry(self, retry_state):
        exception = retry_state.outcome.exception()
        self.log(f"⚠️ API Request Failed: {exception}. Retrying in {retry_state.next_action.sleep}s... (Attempt {retry_state.attempt_number})")

    # --- Robust API Methods with Retry ---
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _get_balance_api(self):
        if not self.client: raise Exception("Client not initialized")
        info = self.client.futures_account_balance()
        for asset in info:
            if asset['asset'] == 'USDT':
                return float(asset['balance'])
        return 0.0

    def get_balance(self):
        if self.mode == 'Live' and self.client:
            try:
                return self._get_balance_api()
            except Exception as e:
                self.log(f"⚠️ Balance Check Error: {e}")
                return 0.0
        return 10000.0 # Paper Money

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _create_order_api(self, **kwargs):
        if not self.client: raise Exception("Client not initialized")
        # For Futures, we usually use futures_create_order
        return self.client.futures_create_order(**kwargs)

    # --- Unified Execution Logic (Synced with Backtest) ---
    def sync_and_execute(self, long_signal, short_signal, current_price, allocation, leverage=1.0, sl=0.03, tp=0.1):
        """
        Main loop hook to sync live state with strategy signals.
        Matches _fast_dca_core_v2 logic from backtest.py.
        """
        symbol = "BTCUSDT"
        tag = f"[{self.mode}]"
        
        # 0. Load Current Position state
        current_dir = 0
        if self.active_trade:
            current_dir = 1 if self.active_trade['side'] == 'LONG' else -1
            
            # --- SL / TP Monitoring ---
            entry = self.active_trade['entry_price']
            side = self.active_trade['side']
            
            # For Paper trading, simulate DCA fill
            if self.mode == 'Paper' and not self.active_trade['dca_filled']:
                dca_p = self.active_trade['dca_price']
                if (side == 'LONG' and current_price <= dca_p) or (side == 'SHORT' and current_price >= dca_p):
                    self.log(f"💰 {tag} DCA 체결 완료 (Avg Price 갱신)")
                    # New Avg Price: (P1 + P2) / 2 assuming equal size
                    new_avg = (entry + dca_p) / 2
                    self.active_trade['entry_price'] = new_avg
                    self.active_trade['amount'] *= 2
                    self.active_trade['dca_filled'] = True
                    entry = new_avg

            # Calculate PnL Pct
            pnl_pct = (current_price - entry) / entry if side == 'LONG' else (entry - current_price) / entry
            
            if pnl_pct <= -sl:
                self.log(f"🩸 {tag} 손절(SL) 트리거 ({pnl_pct*100:.2f}%)")
                self._close_position(symbol, current_price)
                current_dir = 0
            elif pnl_pct >= tp:
                self.log(f"💎 {tag} 익절(TP) 트리거 ({pnl_pct*100:.2f}%)")
                self._close_position(symbol, current_price)
                current_dir = 0

        if current_dir == 0:
            # 1. Determine Target Direction from Strategy
            target_dir = 0
            if long_signal: target_dir = 1
            elif short_signal: target_dir = -1

            # 2. Check for New Entry
            if target_dir != 0:
                self.log(f"🚀 {tag} 신규 진입 감지: {'LONG 🟢' if target_dir==1 else 'SHORT 🔴'}")
                self._open_position(symbol, target_dir, current_price, allocation, leverage)
        else:
             # Check for Signal Reversal (Switching)
             target_dir = 0
             if long_signal: target_dir = 1
             elif short_signal: target_dir = -1
             
             if target_dir != 0 and target_dir != current_dir:
                 self.log(f"🔄 {tag} 시그널 반전 감지 -> 스위칭")
                 self._close_position(symbol, current_price)
                 self._open_position(symbol, target_dir, current_price, allocation, leverage)

    def _open_position(self, symbol, direction, price, allocation, leverage):
        tag = f"[{self.mode}]"
        side = 'BUY' if direction == 1 else 'SELL'
        
        # 1. Market Entry (50% of leverage-adjusted allocation)
        entry_usdt = (allocation * leverage) * 0.5
        
        if self.mode == 'Live':
            try:
                # Set Leverage first
                self.client.futures_change_leverage(symbol=symbol, leverage=int(leverage))
                
                if direction == 1:
                    # For Long, we can use quoteOrderQty (not supported in all futures endpoints, so we use qty)
                    qty = round(entry_usdt / price, 3)
                    order = self._create_order_api(symbol=symbol, side=side, type='MARKET', quantity=qty)
                else:
                    qty = round(entry_usdt / price, 3)
                    order = self._create_order_api(symbol=symbol, side=side, type='MARKET', quantity=qty)
                
                actual_price = float(order.get('avgPrice', price))
                self.log(f"  > ✅ 1차 진입 완료: {side} @ {round(actual_price, 2)}")
            except Exception as e:
                self.log(f"❌ 진입 실패: {e}")
                return
        else:
            actual_price = price
            self.log(f"  > 📝 [Paper] 1차 진입 완료: {side} @ {actual_price}")

        self.active_trade = {
            'side': 'LONG' if direction == 1 else 'SHORT',
            'entry_price': actual_price,
            'amount': entry_usdt,
            'dca_filled': False,
            'dca_price': actual_price * (0.98 if direction == 1 else 1.02)
        }

        # 2. Setup DCA (Limit Order at 2% away)
        dca_price = actual_price * (0.98 if direction == 1 else 1.02)
        dca_qty = entry_usdt / dca_price
        
        if self.mode == 'Live':
            try:
                self._create_order_api(
                    symbol=symbol, side=side, type='LIMIT', timeInForce='GTC',
                    quantity="{:.3f}".format(dca_qty), price="{:.2f}".format(dca_price)
                )
                self.log(f"  > ⏳ 2차 DCA 예약 완료: {round(dca_price, 2)}")
            except Exception as e: self.log(f"⚠️ DCA 예약 실패: {e}")
        else:
            self.log(f"  > 📝 [Paper] 2차 DCA 예약 완료: {dca_price:.2f}")

    def _close_position(self, symbol, price):
        if not self.active_trade: return
        tag = f"[{self.mode}]"
        # Closing order is opposite side
        side = 'SELL' if self.active_trade['side'] == 'LONG' else 'BUY'
        
        if self.mode == 'Live':
            try:
                # Cancel pending orders
                self.client.futures_cancel_all_open_orders(symbol=symbol)
                
                # Fetch position size
                pos = self.client.futures_position_information(symbol=symbol)
                size = 0
                for p in pos:
                    if p['symbol'] == symbol: size = abs(float(p['positionAmt']))
                
                if size > 0:
                    self._create_order_api(symbol=symbol, side=side, type='MARKET', quantity=size)
                    self.log(f"  > ✅ 전량 청산 완료 @ {price}")
                else: self.log("  > ⚠️ 청산할 포지션이 없습니다.")
            except Exception as e: self.log(f"❌ 청산 실패: {e}")
        else:
            self.log(f"  > 📝 [Paper] 전량 청산 완료 @ {price}")
            
        self.active_trade = None

    def execute_live_trade(self, signal, price, amount):
        pass # Legacy compatibility
