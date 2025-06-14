import os
import datetime
import pandas as pd
import numpy as np
import time
import logging
import pyotp
from scipy.signal import argrelextrema
from SmartApi import SmartConnect, smartExceptions

# ======= CONFIGURATION ========
SYMBOL_TOKEN_FILE = 'symbol-token.txt'
HISTORY_YEARS = 2
ANALYSIS_DAYS = 60                # days to analyze
OUTPUT_CSV = 'breakouts.csv'
CACHE_DIR = 'cache'
HOURLY_INTERVAL = 'ONE_HOUR'
DAILY_INTERVAL = 'ONE_DAY'
FIVEMIN_INTERVAL = 'FIVE_MINUTE'
TENMIN_INTERVAL = 'TEN_MINUTE'

# Breakout criteria
MIN_LEVEL_AGE = 30                # days
MIN_TOUCHES = 2
VOL_MULTIPLIER = 1.5
RSI_RANGE = {'bullish': (55, 75), 'bearish': (25, 40)}
CONFIRMATION_CANDLES = 3          # Candles to confirm breakout

# Technical parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOL_AVG_WINDOW = 20
PIVOT_WINDOW_HOURLY = 24          # periods

# Risk management
SL_BUFFER = 0.03   # 3%
TARGET1_PCT = 0.08
TARGET2_PCT = 0.15

# SmartAPI credentials (secure in prod)
API_KEY = '3ZkochvK'
USERNAME = 'D61366376'
PASSWORD = '2299'
TOTP_SECRET = 'B4C2S5V6DUWUP2E4SFVRWA5CGE'
# ==============================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_symbol_tokens(path):
    df = pd.read_csv(path, names=['symbol','token'], dtype=str)
    return {row.symbol: row.token for _, row in df.iterrows() if row.token and row.token != 'NOT_FOUND'}


def init_smartapi():
    smart = SmartConnect(API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    sess = smart.generateSession(USERNAME, PASSWORD, totp)
    if not sess.get('status'):
        raise RuntimeError('SmartAPI login failed')
    smart.generateToken(sess['data']['refreshToken'])
    return smart


def fetch_candles(smart, symbol, token, start_date, end_date, interval):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = f"{token}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    if os.path.exists(cache_file):
        try:
            df_cache = pd.read_pickle(cache_file)
            logger.info(f"[{symbol}] Loaded cache {interval} {start_date}→{end_date}")
            return df_cache
        except Exception as e:
            logger.warning(f"[{symbol}] Cache load failed: {e}")

    params = {
        'exchange': 'NSE',
        'symboltoken': token,
        'interval': interval,
        'fromdate': start_date.strftime('%Y-%m-%d 09:15'),
        'todate':   end_date.strftime('%Y-%m-%d 15:30')
    }
    for attempt in range(3):
        try:
            resp = smart.getCandleData(params)
            if resp.get('data'):
                df = pd.DataFrame(resp['data'], columns=['timestamp','open','high','low','close','volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.to_pickle(cache_file)
                logger.info(f"[{symbol}] Cached {len(df)} rows for {interval}")
                return df
        except smartExceptions.DataException:
            logger.warning(f"[{symbol}] No data for {interval}")
            break
        except Exception as e:
            logger.error(f"[{symbol}] Fetch error {interval}: {e}")
        time.sleep(0.1 * (2 ** attempt))
    return pd.DataFrame()


def compute_rsi(series, period=RSI_PERIOD):
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line


def find_pivots(high, low, window=5):
    max_idx = argrelextrema(high.values, np.greater, order=window)[0]
    min_idx = argrelextrema(low.values, np.less, order=window)[0]
    return high.iloc[max_idx], low.iloc[min_idx]


def calculate_level_strength(df, level, typ, tol=0.015):
    touches = 0
    first_t = None
    for idx, row in df.iterrows():
        price = row['high'] if typ=='resistance' else row['low']
        if abs(price - level) / level <= tol:
            touches += 1
            first_t = first_t or idx
    if touches < MIN_TOUCHES:
        return 0, None
    age = (df.index.max().date() - first_t.date()).days
    score = min(touches * 0.3 + min(age/30, 2) * 0.7, 1.0)
    return score, age


def find_strong_levels(df, pivot_window=5):
    recent = df if len(df) <= VOL_AVG_WINDOW*6 else df.iloc[-VOL_AVG_WINDOW*6:]
    piv_res, piv_sup = find_pivots(recent['high'], recent['low'], window=pivot_window)
    levels = {'res': [], 'sup': []}
    current = df['close'].iloc[-1]
    for lvl in np.unique(piv_res):
        score, age = calculate_level_strength(df, lvl, 'resistance')
        if score >= 0.4 and lvl > current and age >= MIN_LEVEL_AGE:
            levels['res'].append({'level': lvl, 'score': score, 'age': age})
    for lvl in np.unique(piv_sup):
        score, age = calculate_level_strength(df, lvl, 'support')
        if score >= 0.4 and lvl < current and age >= MIN_LEVEL_AGE:
            levels['sup'].append({'level': lvl, 'score': score, 'age': age})
    levels['res'].sort(key=lambda x: x['level'])
    levels['sup'].sort(key=lambda x: x['level'], reverse=True)
    return levels

def round_target(price):
    if price < 100:
        return round(price, 2)
    elif price < 500:
        return round(price, 1)
    else:
        return round(price)  # No decimals

def analyze_stock(smart, symbol, token):
    logger.info(f"Analyzing {symbol}")
    today = datetime.date.today()
    start_hist = today - datetime.timedelta(days=HISTORY_YEARS*365)
    start_intra = today - datetime.timedelta(days=ANALYSIS_DAYS)

    # 1) Pivot data
    df_hist = fetch_candles(smart, symbol, token, start_hist, today, HOURLY_INTERVAL)
    if df_hist.empty:
        df_hist = fetch_candles(smart, symbol, token, start_hist, today, DAILY_INTERVAL)
    if df_hist.empty or len(df_hist) < 50:
        return []

    # 2) Intraday bulk
    df_intra = fetch_candles(smart, symbol, token, start_intra, today, FIVEMIN_INTERVAL)
    if df_intra.empty:
        df_intra = fetch_candles(smart, symbol, token, start_intra, today, TENMIN_INTERVAL)
    if df_intra.empty:
        return []
    df_intra['rsi'] = compute_rsi(df_intra['close'])
    df_intra['macd'], df_intra['signal'] = compute_macd(df_intra['close'])
    df_intra['vol_avg'] = df_intra['volume'].rolling(VOL_AVG_WINDOW).mean()

    results = []
    for i in range(1, ANALYSIS_DAYS+1):
        day = today - datetime.timedelta(days=i)
        piv_df = df_hist[df_hist.index.date < day]
        if len(piv_df) < 50:
            continue
        levels = find_strong_levels(piv_df, pivot_window=PIVOT_WINDOW_HOURLY)
        if not levels['res'] and not levels['sup']:
            continue

        day_df = df_intra[df_intra.index.date == day]
        if day_df.empty:
            continue

        for ts, row in day_df.iterrows():
            if np.isnan(row['vol_avg']) or np.isnan(row['rsi']):
                continue
            action, lvl = None, None
            # BUY check
            if levels['res']:
                res_lvls = [l['level'] for l in levels['res'] if l['level'] < row['close']]
                if res_lvls and row['volume'] > VOL_MULTIPLIER * row['vol_avg'] and \
                   RSI_RANGE['bullish'][0] <= row['rsi'] <= RSI_RANGE['bullish'][1] and \
                   row['macd'] > row['signal']:
                    action = 'BUY'
                    lvl = {'level': max(res_lvls), 'score': None, 'age': None}
            # SELL check
            if not action and levels['sup']:
                sup_lvls = [l['level'] for l in levels['sup'] if l['level'] > row['close']]
                if sup_lvls and row['volume'] > VOL_MULTIPLIER * row['vol_avg'] and \
                   RSI_RANGE['bearish'][0] <= row['rsi'] <= RSI_RANGE['bearish'][1] and \
                   row['macd'] < row['signal']:
                    action = 'SELL'
                    lvl = {'level': min(sup_lvls), 'score': None, 'age': None}
            if not action:
                continue
            bp = round_target(lvl['level'])

            sl_price = bp * (1 - SL_BUFFER) if action == 'BUY' else bp * (1 + SL_BUFFER)
            # targets from pivots
            piv_list = levels['res'] if action == 'BUY' else levels['sup']
            pts = [l['level'] for l in piv_list if (l['level'] > bp if action == 'BUY' else l['level'] < bp)]
            t_1 = pts[0] if len(pts) > 0 else bp * (1 + (TARGET1_PCT if action == 'BUY' else -TARGET1_PCT))
            t_2 = pts[1] if len(pts) > 1 else bp * (1 + (TARGET2_PCT if action == 'BUY' else -TARGET2_PCT))

            if action=='BUY':
                sorted_targets = sorted([bp * 1.06, t_1, t_2])
                t1 = round_target(sorted_targets[0])
                t2 = round_target(sorted_targets[1])
                t3 = round_target(sorted_targets[2])
            else:
                sorted_targets = sorted([bp * (1-0.06), t_1, t_2], reverse=True)
                t1 = round_target(sorted_targets[0])
                t2 = round_target(sorted_targets[1])
                t3 = round_target(sorted_targets[2])

            results.append({
                'date': day.strftime('%Y-%m-%d'),
                'time': ts.strftime('%H:%M'),
                'symbol': symbol,
                'token': token,
                'action': action,
                'breakout_level': bp,
                'stop_loss': sl_price,
                'target1': t1,
                'target2': t2,
                'target3': t3,
                'rsi': row['rsi'],
                'macd': row['macd'],
                'volume': row['volume'],
                'level_score': lvl.get('score'),
                'level_age': lvl.get('age'),
                'close': row['close']
            })
            break

    return results

def main():
    smart = init_smartapi()
    full_map = load_symbol_tokens(SYMBOL_TOKEN_FILE)
    all_results = []
    total = len(full_map)
    for i, (sym, tok) in enumerate(full_map.items(), 1):
        if i > 201:
            break
        logger.info(f"Processing {sym} ({i}/{total})")
        picks = analyze_stock(smart, sym, tok)
        all_results.extend(picks)
        time.sleep(0.1)

    df = pd.DataFrame(all_results)
    # only keep BUY trades
    df = df[df['action'] == 'BUY']
    # ensure date/time are proper types for sorting
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = pd.to_datetime(df['time'], format='%H:%M').dt.time
    # sort by symbol, then date (ascending), then time
    df = df.sort_values(['symbol','date','time'])
    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == '__main__':
    main()