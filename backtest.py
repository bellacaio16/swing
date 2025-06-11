import os
import datetime
import pandas as pd
import numpy as np
import time
import logging
import pyotp
from SmartApi import SmartConnect, smartExceptions

# ======= CONFIGURATION ========
BREAKOUT_CSV         = 'intraday_breakouts.csv'
SWING_BACKTEST_CSV   = 'swing_backtest_results.csv'
CACHE_DIR            = 'cache_daily'
DAILY_INTERVAL       = 'ONE_DAY'
MAX_HOLD_DAYS        = 10
MAX_CAPITAL_PER_TRADE = 35000  # ₹35k max per trade

# SmartAPI credentials
API_KEY     = '3ZkochvK'
USERNAME    = 'D61366376'
PASSWORD    = '2299'
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
    cache_file = os.path.join(CACHE_DIR, f"{token}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl")
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)
    
    params = {
        'exchange': 'NSE',
        'symboltoken': token,
        'interval': DAILY_INTERVAL,
        'fromdate': start_date.strftime('%Y-%m-%d 00:00'),
        'todate': end_date.strftime('%Y-%m-%d 23:59')
    }
    resp = smart.getCandleData(params)
    if resp.get('data'):
        df = pd.DataFrame(resp['data'], columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
        df.set_index('timestamp', inplace=True)
        df.to_pickle(cache_file)
        return df
    return pd.DataFrame()


def swing_backtest_trade(smart, trade):
    entry_date = datetime.datetime.strptime(trade['entry_date'], '%Y-%m-%d').date()
    token = trade['token']
    action = trade['action']
    breakout = trade['breakout_level']
    sl = trade['stop_loss']
    t1 = trade['target1']
    t2 = trade['target2']

    position_size = int(MAX_CAPITAL_PER_TRADE // breakout) if breakout > 0 else 0
    capital_used = position_size * breakout

    start = entry_date + datetime.timedelta(days=1)
    end = entry_date + datetime.timedelta(days=MAX_HOLD_DAYS)
    df = fetch_daily_candles(smart, token, start, end)
    if df.empty or position_size <= 0:
        return None

    hit_t1 = False
    exit_price = None
    exit_date = None
    result = None

    for day, row in df.iterrows():
        high = row['high']
        low = row['low']
        if action == 'BUY':
            if not hit_t1:
                if low <= sl:
                    exit_price, exit_date, result = sl, day, 'SL'
                    break
                if high >= t1:
                    hit_t1 = True
                if high >= t2:
                    exit_price, exit_date, result = t2, day, 'T2'
                    break
            else:
                if low <= breakout:
                    exit_price, exit_date, result = breakout, day, 'T1_SL'
                    break
                if high >= t2:
                    exit_price, exit_date, result = t2, day, 'T2'
                    break
        else:
            if not hit_t1:
                if high >= sl:
                    exit_price, exit_date, result = sl, day, 'SL'
                    break
                if low <= t1:
                    hit_t1 = True
                if low <= t2:
                    exit_price, exit_date, result = t2, day, 'T2'
                    break
            else:
                if high >= breakout:
                    exit_price, exit_date, result = breakout, day, 'T1_SL'
                    break
                if low <= t2:
                    exit_price, exit_date, result = t2, day, 'T2'
                    break
    else:
        exit_date = df.index[-1]
        exit_price = df['close'].iloc[-1]
        result = 'MKT'

    pnl_per_share = (exit_price - breakout) if action == 'BUY' else (breakout - exit_price)
    pnl_total = pnl_per_share * position_size

    return {
        **trade,
        'position_size': position_size,
        'capital_used': capital_used,
        'exit_date': exit_date.strftime('%Y-%m-%d'),
        'exit_price': exit_price,
        'result': result,
        'pnl_per_share': pnl_per_share,
        'pnl_total': pnl_total,
        'holding_days': (exit_date - start + datetime.timedelta(days=1)).days
    }


def main():
    smart = init_smartapi()
    trades_df = pd.read_csv(BREAKOUT_CSV, dtype={'date': str, 'token': str, 'action': str})

    if 'time' not in trades_df.columns:
        trades_df['time'] = '09:15:00'  # default if missing

    trades_df['datetime'] = pd.to_datetime(trades_df['date'] + ' ' + trades_df['time'])
    trades_df = trades_df.sort_values(by='datetime')

    results = []
    ongoing_positions = {}

    for _, row in trades_df.iterrows():
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
            'entry_date': row['date']
        }

        if symbol in ongoing_positions:
            held = ongoing_positions[symbol]
            held_entry = datetime.datetime.strptime(held['entry_date'], '%Y-%m-%d').date()
            if current_date <= held_entry + datetime.timedelta(days=MAX_HOLD_DAYS):
                # Still holding → update SL/targets
                if action == 'BUY':
                    held['target1'] = max(held['target1'], new_trade['target1'])
                    held['target2'] = max(held['target2'], new_trade['target2'])
                    held['stop_loss'] = min(held['stop_loss'], new_trade['stop_loss'])
                elif action == 'SELL':
                    held['target1'] = min(held['target1'], new_trade['target1'])
                    held['target2'] = min(held['target2'], new_trade['target2'])
                    held['stop_loss'] = max(held['stop_loss'], new_trade['stop_loss'])
                continue
            else:
                # Exit and replace
                res = swing_backtest_trade(smart, held)
                if res:
                    results.append(res)
                del ongoing_positions[symbol]


        ongoing_positions[symbol] = new_trade
        time.sleep(0.1)

    # Final flush of all held positions
    for sym, trade in ongoing_positions.items():
        res = swing_backtest_trade(smart, trade)
        if res:
            results.append(res)

    pd.DataFrame(results).to_csv(SWING_BACKTEST_CSV, index=False)


if __name__ == '__main__':
    main()
