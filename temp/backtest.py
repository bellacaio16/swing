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
MAX_HOLD_DAYS = 10
MAX_CAPITAL_PER_TRADE = 10000  # ₹10k max per trade

# Global capital pool
INITIAL_CAPITAL = 1000000  # ₹200k total

# SmartAPI credentials
API_KEY = '3ZkochvK'
USERNAME = 'D61366376'
PASSWORD = '2299'
TOTP_SECRET = 'B4C2S5V6DUWUP2E4SFVRWA5CGE'
# ==============================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def init_smartapi(retries=3, backoff=1, timeout=20):
    """
    Initialize SmartAPI session with retry and custom timeout.
    """
    for attempt in range(1, retries + 1):
        try:
            smart = SmartConnect(API_KEY, timeout=timeout)
            totp = pyotp.TOTP(TOTP_SECRET).now()
            sess = smart.generateSession(USERNAME, PASSWORD, totp)
            if not sess.get('status'):
                raise RuntimeError('SmartAPI login failed: ' + str(sess))
            smart.generateToken(sess['data']['refreshToken'])
            logger.info(f"Logged in successfully on attempt {attempt}")
            return smart
        except Exception as e:
            logger.warning(f"Attempt {attempt} login error: {e}")
            if attempt < retries:
                time.sleep(backoff * (2 ** (attempt - 1)))
            else:
                logger.error("All login attempts failed. Exiting.")
                raise


def fetch_daily_candles(smart, token, start_date, end_date):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(
        CACHE_DIR,
        f"{token}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
    )
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)

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
            time.sleep(2 ** attempt * 0.5)

    if resp.get('data'):
        df = pd.DataFrame(resp['data'],
                          columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
        df.set_index('timestamp', inplace=True)
        df.to_pickle(cache_file)
        return df

    return pd.DataFrame()


def swing_backtest_trade(smart, trade, end_date=None):
    entry_date = trade['entry_date']
    token = trade['token']
    action = trade['action']
    breakout = trade['breakout_level']
    sl = trade['stop_loss']
    t1 = trade['target1']
    t2 = trade['target2']
    t3 = trade['target3']

    # use passed-in position size and capital
    position_size = trade.get('position_size')
    capital_used = trade.get('capital_used')
    if position_size is None or capital_used is None:
        return None

    # Simulation period
    start = entry_date + timedelta(days=1)
    if end_date:
        end = end_date
    else:
        end = entry_date + timedelta(days=MAX_HOLD_DAYS)

    if end < start:
        return {
            'symbol': trade['symbol'],
            'token': token,
            'action': action,
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'position_size': position_size,
            'capital_used': capital_used,
            'exit_date': entry_date.strftime('%Y-%m-%d'),
            'exit_price': breakout,
            'result': 'SAME_DAY_EXIT',
            'pnl_per_share': 0,
            'pnl_total': 0,
            'holding_days': 0
        }

    df = fetch_daily_candles(smart, token, start, end)
    if df.empty:
        return None

    hit_t1 = hit_t2 = False
    exit_price = exit_date = result = None

    for day, row in df.iterrows():
        high, low = row['high'], row['low']
        if action == 'BUY':
            if not hit_t1:
                if low <= sl:
                    exit_price, exit_date, result = sl, day, 'SL'; break
                if high >= t1: hit_t1 = True
                if high >= t2: hit_t2 = True
                if high >= t3:
                    exit_price, exit_date, result = t3, day, 'T3'; break
            elif hit_t1 and not hit_t2:
                if low <= breakout:
                    exit_price, exit_date, result = breakout, day, 'T1_SL'; break
                if high >= t2: hit_t2 = True
                if high >= t3:
                    exit_price, exit_date, result = t3, day, 'T3'; break
            else:
                if low <= t1:
                    exit_price, exit_date, result = t1, day, 'T2_SL'; break
                if high >= t3:
                    exit_price, exit_date, result = t3, day, 'T3'; break

    if exit_price is None:
        exit_date = df.index[-1]
        exit_price = df['close'].iloc[-1]
        result = 'MKT'

    pnl_per_share = (exit_price - breakout) if action == 'BUY' else (breakout - exit_price)
    pnl_total = pnl_per_share * position_size
    holding_days = (exit_date - entry_date).days

    return {
        'symbol': trade['symbol'],
        'token': token,
        'action': action,
        'entry_date': entry_date.strftime('%Y-%m-%d'),
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
    trades_df = pd.read_csv(BREAKOUT_CSV, dtype=str)

    # Ensure we have a time column
    if 'time' not in trades_df.columns:
        trades_df['time'] = '09:15:00'

    # Parse date strings (day-first) into date objects
    trades_df['date'] = pd.to_datetime(trades_df['date'], dayfirst=True).dt.date
    trades_df['datetime'] = pd.to_datetime(trades_df['date'].astype(str) + ' ' + trades_df['time'])
    trades_df = trades_df.sort_values(by='datetime')

    results = []
    ongoing_positions = {}
    total_capital = INITIAL_CAPITAL
    logger.info(f"Starting backtest with total capital = ₹{total_capital:.0f}")

    for _, row in trades_df.iterrows():
        symbol = row['symbol']
        token = row['token']
        action = row['action']
        current_date = row['date']

        # If already holding and within max days, just merge SL/targets
        if symbol in ongoing_positions:
            held = ongoing_positions[symbol]
            held_entry = held['entry_date']
            if current_date <= held_entry + timedelta(days=MAX_HOLD_DAYS):
                held['stop_loss'] = max(held['stop_loss'], float(row['stop_loss']))
                held.update({
                    'target1': float(row['target1']),
                    'target2': float(row['target2']),
                    'target3': float(row['target3'])
                })
                continue
            else:
                res = swing_backtest_trade(smart, held)
                if res:
                    total_capital += res['capital_used'] + res['pnl_total']
                    logger.info(f"🔄 Exited {res['symbol']} @ {res['exit_price']} on {res['exit_date']}: P/L=₹{res['pnl_total']:.0f}, free_capital=₹{total_capital:.0f}")
                    results.append(res)
                del ongoing_positions[symbol]

        # Prepare new trade
        breakout = float(row['breakout_level'])
        position_size = int(MAX_CAPITAL_PER_TRADE // breakout)
        capital_needed = position_size * breakout
        if total_capital < capital_needed:
            logger.info(f"⏭ Skipping {symbol} @ {breakout} – only ₹{total_capital:.0f} free")
            continue

        total_capital -= capital_needed
        new_trade = {
            'symbol': symbol,
            'token': token,
            'action': action,
            'breakout_level': breakout,
            'stop_loss': float(row['stop_loss']),
            'target1': float(row['target1']),
            'target2': float(row['target2']),
            'target3': float(row['target3']),
            'entry_date': current_date,
            'position_size': position_size,
            'capital_used': capital_needed
        }
        ongoing_positions[symbol] = new_trade
        logger.info(f"✅ Opened {symbol}: size={position_size}, cost=₹{capital_needed:.0f}, free_capital=₹{total_capital:.0f}")
        time.sleep(0.1)

    for trade in ongoing_positions.values():
        res = swing_backtest_trade(smart, trade)
        if not res:
            continue
        total_capital += res['capital_used'] + res['pnl_total']
        logger.info(f"🔄 Exited {res['symbol']} @ {res['exit_price']} on {res['exit_date']}: P/L=₹{res['pnl_total']:.0f}, free_capital=₹{total_capital:.0f}")
        results.append(res)

    if results:
        pd.DataFrame(results).to_csv(SWING_BACKTEST_CSV, index=False)
        logger.info(f"Backtest complete. Final free capital = ₹{total_capital:.0f}. Results saved to {SWING_BACKTEST_CSV}")
    else:
        logger.warning("No backtest results to save")

if __name__ == '__main__':
    main()
