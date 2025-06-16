import os
import datetime
import pandas as pd
import numpy as np
from datetime import timedelta, date
import time
import logging
import pyotp
from SmartApi import SmartConnect, smartExceptions

# ======= CONFIGURATION ========
BREAKOUT_CSV = 'breakouts.csv'
SWING_BACKTEST_CSV = 'swing_backtest_results.csv'
CACHE_DIR = 'cache_daily'
DAILY_INTERVAL = 'ONE_DAY'
MAX_HOLD_DAYS = 7   # Max trading days to hold
MAX_CAPITAL_PER_TRADE = 6500  # ₹7k max per trade
TOTAL_CAPITAL_START = 200000  # ₹2,00,000 initial capital
# SmartAPI credentials
API_KEY = '4QWgqJPV'
USERNAME = 'D61366376'
PASSWORD = '2299'
TOTP_SECRET = 'B4C2S5V6DUWUP2E4SFVRWA5CGE'
# ==============================

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
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

    span = (end_date - start_date).days
    if span > 30:
        parts = []
        d = start_date
        while d <= end_date:
            chunk_end = min(d + timedelta(days=30), end_date)
            part = fetch_daily_candles(smart, token, d, chunk_end)
            if not part.empty:
                parts.append(part)
            d = chunk_end + timedelta(days=1)
        if parts:
            df = pd.concat(parts).sort_index()
            df.to_pickle(cache_file)
            return df
        return pd.DataFrame()

    params = {
        'exchange': 'NSE',
        'symboltoken': token,
        'interval': DAILY_INTERVAL,
        'fromdate': start_date.strftime('%Y-%m-%d 00:00'),
        'todate':   end_date.strftime('%Y-%m-%d 23:59')
    }
    for attempt in range(3):
        try:
            resp = smart.getCandleData(params)
            print("fetched")
            break
        except Exception as e:
            logger.warning(f" Atempt = {attempt+1}   {token} fetch error: {e}")
            time.sleep(0.5 * 2 ** attempt)
    else:
        return pd.DataFrame()

    data = resp.get('data') or []
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
    df.set_index('timestamp', inplace=True)
    df.to_pickle(cache_file)
    return df

def main():
    smart = init_smartapi()
    df = pd.read_csv(BREAKOUT_CSV, dtype=str)
    df['time']   = df.get('time',  '09:15:00')
    df['date']   = pd.to_datetime(df['date'], dayfirst=True).dt.date
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    df = df.sort_values('datetime')
    trades_by_date = df.groupby('date')

    start_date = df['date'].min()
    last_signal = df['date'].max()
    total_capital = TOTAL_CAPITAL_START
    ongoing = {}      # symbol -> position dict
    results = []

    # 1. Loop from first through last signal date
    current = start_date
    while current <= last_signal:
        # (a) Update existing positions
        for sym in list(ongoing):
            pos = ongoing[sym]
            day_df = fetch_daily_candles(smart, pos['token'], current, current)
            if day_df.empty:
                continue  # non-trading day

            high, low, close = day_df['high'].iloc[0], day_df['low'].iloc[0], day_df['close'].iloc[0]
            pos['days_held'] += 1

            # flag profit targets
            if not pos['hit1'] and high >= pos['targets'][0]:
                pos['hit1'] = True
            if not pos['hit2'] and high >= pos['targets'][1]:
                pos['hit2'] = True

            exit_flag = False
            # check exits in proper order
            if not pos['hit1'] and low <= pos['sl']:
                price, reason = pos['sl'], 'SL'; exit_flag = True
            elif pos['hit1'] and not pos['hit2'] and low <= pos['breakout']:
                price, reason = pos['breakout'], 'T1_SL'; exit_flag = True
            elif pos['hit2'] and low <= pos['targets'][0]:
                price, reason = pos['targets'][0], 'T2_SL'; exit_flag = True
            elif high >= pos['targets'][2]:
                price, reason = pos['targets'][2], 'T3'; exit_flag = True
            elif pos['days_held'] > MAX_HOLD_DAYS:
                price, reason = close, 'MKT'; exit_flag = True

            if exit_flag:
                pnl = (price - pos['breakout']) * pos['size']
                total_capital += pos['capital'] + pnl
                results.append({
                    **pos,
                    'exit_date':     current.strftime('%Y-%m-%d'),
                    'exit_price':    price,
                    'result':        reason,
                    'pnl_total':     pnl,
                    'remaining_capital': total_capital
                })
                print("EXIT")
                del ongoing[sym]

        # (b) Open or replace trades for today
        if current in trades_by_date.groups:
            for _, row in trades_by_date.get_group(current).iterrows():
                sym      = row['symbol']
                breakout = float(row['breakout_level'])
                size     = int(MAX_CAPITAL_PER_TRADE / breakout)

                # skip zero-size
                if size < 1:
                    logger.info(f"Skipping {sym}: breakout too high for minimum size")
                    continue

                cap_used = size * breakout

                # replace if already holding
                if sym in ongoing:
                    pos = ongoing.pop(sym)
                    day_df = fetch_daily_candles(smart, pos['token'], current, current)
                    if not day_df.empty:
                        open_price = day_df['open'].iloc[0]
                        pnl = (open_price - pos['breakout']) * pos['size']
                        total_capital += pos['capital'] + pnl
                        results.append({
                            **pos,
                            'exit_date':     current.strftime('%Y-%m-%d'),
                            'exit_price':    open_price,
                            'result':        'REPLACE',
                            'pnl_total':     pnl,
                            'remaining_capital': total_capital
                        })

                # then new entry if capital allows
                if total_capital >= cap_used:
                    total_capital -= cap_used
                    ongoing[sym] = {
                        'symbol':       sym,
                        'entry_date':   current.strftime('%Y-%m-%d'),
                        'breakout':     breakout,
                        'sl':           float(row['stop_loss']),
                        'targets':      (
                                           float(row['target1']),
                                           float(row['target2']),
                                           float(row['target3'])
                                       ),
                        'token':        row['token'],
                        'size':         size,
                        'capital':      cap_used,
                        'hit1':         False,
                        'hit2':         False,
                        'days_held':    0
                    }
                else:
                    logger.info(
                        f"SKIPPED for {sym} on {current}: need {cap_used}, have {total_capital}"
                    )

        current += timedelta(days=1)

    # 2. Exit any remaining positions on the last signal date
    for sym, pos in list(ongoing.items()):
        day_df = fetch_daily_candles(smart, pos['token'], last_signal, last_signal)
        if not day_df.empty:
            close_price = day_df['close'].iloc[0]
        else:
            close_price = pos['breakout']  # fallback
        pnl = (close_price - pos['breakout']) * pos['size']
        total_capital += pos['capital'] + pnl
        results.append({
            **pos,
            'exit_date':        last_signal.strftime('%Y-%m-%d'),
            'exit_price':       close_price,
            'result':           'END',
            'pnl_total':        pnl,
            'remaining_capital': total_capital
        })
    ongoing.clear()

    # Write out results
    if results:
        pd.DataFrame(results).to_csv(SWING_BACKTEST_CSV, index=False)
    else:
        logger.warning("No backtest results to save")

if __name__ == '__main__':
    main()
