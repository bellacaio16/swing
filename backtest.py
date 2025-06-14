import os
import datetime
import pandas as pd
import numpy as np
from datetime import timedelta
import time
import logging
import pyotp
from SmartApi import SmartConnect, smartExceptions

# ======= CONFIGURATION ========
BREAKOUT_CSV = 'breakouts.csv'
SWING_BACKTEST_CSV = 'swing_backtest_results.csv'
CACHE_DIR = 'cache_daily'
DAILY_INTERVAL = 'ONE_DAY'
MAX_HOLD_DAYS = 10
MAX_CAPITAL_PER_TRADE = 10000  # ₹35k max per trade

# SmartAPI credentials
API_KEY = '3ZkochvK'
USERNAME = 'D61366376'
PASSWORD = '2299'
TOTP_SECRET = 'B4C2S5V6DUWUP2E4SFVRWA5CGE'
# ==============================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_smartapi():
    smart = SmartConnect(API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    sess = smart.generateSession(USERNAME, PASSWORD, totp)
    if not sess.get('status'):
        raise RuntimeError('SmartAPI login failed')
    smart.generateToken(sess['data']['refreshToken'])
    return smart


def fetch_daily_candles(smart, token, start_date, end_date):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(
        CACHE_DIR,
        f"{token}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
    )
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)

    # If the span is larger than 30 days, split into 30-day chunks:
    span_days = (end_date - start_date).days
    if span_days > 30:
        parts = []
        chunk_start = start_date
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=30), end_date)
            part_df = fetch_daily_candles(smart, token, chunk_start, chunk_end)
            if not part_df.empty:
                parts.append(part_df)
            chunk_start = chunk_end + timedelta(days=1)
        if parts:
            df = pd.concat(parts).sort_index()
            df.to_pickle(cache_file)
            return df
        else:
            return pd.DataFrame()

    # Otherwise, try up to 3 times with exponential back‑off:
    params = {
        'exchange': 'NSE',
        'symboltoken': token,
        'interval': DAILY_INTERVAL,
        'fromdate': start_date.strftime('%Y-%m-%d 00:00'),
        'todate':   end_date.strftime('%Y-%m-%d 23:59')
    }
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = smart.getCandleData(params)
            break
        except Exception as e:
            logger.warning(f"Timeout on getCandleData (attempt {attempt}): {e}")
            if attempt == max_retries:
                logger.error(f"All retries failed for {token} {start_date}–{end_date}")
                return pd.DataFrame()
            time.sleep(2 ** attempt * 0.5)  # 1s, 2s, 4s back‑off

    if resp.get('data'):
        df = pd.DataFrame(resp['data'],
                          columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
        df.set_index('timestamp', inplace=True)
        df.to_pickle(cache_file)
        return df

    return pd.DataFrame()

def swing_backtest_trade(smart, trade, end_date=None):
    """Backtests a trade with optional early exit date"""
    entry_date = datetime.datetime.strptime(trade['entry_date'], '%Y-%m-%d').date()
    token = trade['token']
    action = trade['action']
    breakout = trade['breakout_level']
    sl = trade['stop_loss']
    t1 = trade['target1']
    t2 = trade['target2']
    t3 = trade['target3']

    # log---------------
    print(token + " " + action + "             ==>> EXIT! ")

    # Calculate position size
    if breakout > 0:
        position_size = int(MAX_CAPITAL_PER_TRADE // breakout)
        capital_used = position_size * breakout
    else:
        return None

    # Determine simulation period
    start = entry_date + datetime.timedelta(days=1)
    if end_date:
        end = end_date
    else:
        end = entry_date + datetime.timedelta(days=MAX_HOLD_DAYS)
    
    # Handle cases where end date is before start date
    if end < start:
        return {
            **trade,
            'position_size': position_size,
            'capital_used': capital_used,
            'exit_date': entry_date.strftime('%Y-%m-%d'),
            'exit_price': breakout,
            'result': 'SAME_DAY_EXIT',
            'pnl_per_share': 0,
            'pnl_total': 0,
            'holding_days': 0
        }

    # Fetch price data
    df = fetch_daily_candles(smart, token, start, end)
    if df.empty:
        return None

    # Initialize state variables
    hit_t1 = False
    hit_t2 = False
    exit_price = None
    exit_date = None
    result = None

    # Main simulation loop
    for day, row in df.iterrows():
        high = row['high']
        low = row['low']
        close = row['close']
        
        if action == 'BUY':
            # t1 ni hua hit
            if not hit_t1:
                if low <= sl:
                    exit_price = sl
                    exit_date = day
                    result = 'SL'
                    break
                if high >= t1:
                    hit_t1 = True
                if high >= t2:
                    hit_t2 = True
                if high >= t3:
                    exit_price = t3
                    exit_date = day
                    result = 'T3'
                    break
            if hit_t1 and not hit_t2:
                if low <= breakout:  # Trail SL to breakout
                    exit_price = breakout
                    exit_date = day
                    result = 'T1_SL'
                    break
                if high >= t2:
                    hit_t2 = True
                if high >= t3:
                    exit_price = t3
                    exit_date = day
                    result = 'T3'
                    break
            if hit_t1 and hit_t2:
                if low <= t1:  # Trail SL to T1
                    exit_price = t1
                    exit_date = day
                    result = 'T2_SL'
                    break
                if high >= t3:
                    exit_price = t3
                    exit_date = day
                    result = 'T3'
                    break
        else:     # SELL
            # t1 ni hua hit
            if not hit_t1:
                if high >= sl:
                    exit_price = sl
                    exit_date = day
                    result = 'SL'
                    break
                if low <= t1:
                    hit_t1 = True
                if low <= t2:
                    hit_t2 = True
                if low <= t3:
                    exit_price = t3
                    exit_date = day
                    result = 'T3'
                    break
            if hit_t1 and not hit_t2:
                if high >= breakout:  # Trail SL to breakout
                    exit_price = breakout
                    exit_date = day
                    result = 'T1_SL'
                    break
                if low <= t2:
                    hit_t2 = True
                if low <= t3:
                    exit_price = t3
                    exit_date = day
                    result = 'T3'
                    break
            if hit_t1 and hit_t2:
                if high >= t1:  # Trail SL to T1
                    exit_price = t1
                    exit_date = day
                    result = 'T2_SL'
                    break
                if low <= t3:
                    exit_price = t3
                    exit_date = day
                    result = 'T3'
                    break

    # Handle positions that weren't exited by conditions
    if exit_price is None:
        exit_date = df.index[-1]
        exit_price = df['close'].iloc[-1]
        result = 'MKT'

    # Calculate P&L and holding period
    pnl_per_share = (exit_price - breakout) if action == 'BUY' else (breakout - exit_price)
    pnl_total = pnl_per_share * position_size
    holding_days = (exit_date - entry_date).days

    return {
        **trade,
        'position_size': position_size,
        'capital_used': capital_used,
        'exit_date': exit_date.strftime('%Y-%m-%d'),
        'exit_price': exit_price,
        'result': result,
        'pnl_per_share': pnl_per_share,
        'pnl_total': pnl_total,
        'holding_days': holding_days
    }

def main():
    smart = init_smartapi()
    trades_df = pd.read_csv(BREAKOUT_CSV, dtype={'date': str, 'token': str, 'action': str})

    # Preprocess data
    if 'time' not in trades_df.columns:
        trades_df['time'] = '09:15:00'
    trades_df['datetime'] = pd.to_datetime(trades_df['date'] + ' ' + trades_df['time'])
    trades_df = trades_df.sort_values(by='datetime')

    results = []
    ongoing_positions = {}

    for i, (_, row) in enumerate(trades_df.iterrows()):
        symbol = row['symbol']
        token = row['token']
        action = row['action']
        current_date = row['datetime'].date()

        new_trade = {
            'symbol': symbol,
            'token': token,
            'action': action,
            'breakout_level': row['breakout_level'],
            'stop_loss': row['stop_loss'],
            'target1': row['target1'],
            'target2': row['target2'],
            'target3': row['target3'],
            'entry_date': row['date']
        }
        print(token + " " + action + f" -> Found! -> {i} " )
        # Handle existing positions
        if symbol in ongoing_positions:
            held = ongoing_positions[symbol]
            held_entry = datetime.datetime.strptime(held['entry_date'], '%Y-%m-%d').date()
            
            # Check if still within max hold period and same action
            if (current_date <= held_entry + datetime.timedelta(days=MAX_HOLD_DAYS)) and (action == held['action']):
                # Update existing position - same action

                if action == 'BUY':
                    held['target1'] = new_trade['target1']
                    held['target2'] = new_trade['target2']
                    held['target3'] = new_trade['target3']
                    held['stop_loss'] = max(held['stop_loss'], new_trade['stop_loss'])
                else:  # SELL
                    held['target1'] = new_trade['target1']
                    held['target2'] = new_trade['target2']
                    held['target3'] = new_trade['target3']
                    held['stop_loss'] = min(held['stop_loss'], new_trade['stop_loss'])
                
                # Keep the position with updated parameters
                ongoing_positions[symbol] = held
                continue
            else:
                # Exit existing position (opposite action or beyond max hold days)
                # Use current date as early exit date
                res = swing_backtest_trade(smart, held)
                if res:
                    results.append(res)
                del ongoing_positions[symbol]
        
        # Add new trade to positions
        ongoing_positions[symbol] = new_trade
        time.sleep(0.1)  # Rate limit protection

    # Process any remaining positions
    for sym, trade in ongoing_positions.items():
        res = swing_backtest_trade(smart, trade)
        if res:
            results.append(res)

    # Save results
    if results:
        pd.DataFrame(results).to_csv(SWING_BACKTEST_CSV, index=False)
    else:
        logger.warning("No backtest results to save")

if __name__ == '__main__':
    main()
