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
ANALYSIS_DAYS = 70                # days to analyze
OUTPUT_CSV = 'breakouts.csv'
CACHE_DIR = 'cache'
HOURLY_INTERVAL = 'ONE_HOUR'
DAILY_INTERVAL = 'ONE_DAY'
FIVEMIN_INTERVAL = 'FIVE_MINUTE'
TENMIN_INTERVAL = 'TEN_MINUTE'

# Testing mode: if True, uses TEST_DATE instead of today's date
TEST_MODE = True
TEST_DATE = datetime.date(2025, 6, 20)  # fixed date for consistent caching

# Breakout criteria
MIN_LEVEL_AGE = 30                # days
MIN_TOUCHES = 3
VOL_MULTIPLIER = 1.5
RSI_RANGE = {'bullish': (60, 70)}  # only bullish BUY
CONFIRMATION_CANDLES = 1          # Candles to confirm breakout

# Technical parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOL_AVG_WINDOW = 20
PIVOT_WINDOW_HOURLY = 24          # periods for pivot identification
CLUSTER_TOL = 0.01               # 1% clustering tolerance

# Risk management
SL_BUFFER = 0.005   # 0.5% buffer
TARGET1_PCT = 0.08
TARGET2_PCT = 0.15
TARGET3_PCT = 0.25

# SmartAPI credentials
API_KEY = '4QWgqJPV'
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

def fetch_candles(smart, symbol, token, start_date, end_date, interval, max_attempts=3):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{token}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.exists(cache_file):
        try:
            df = pd.read_pickle(cache_file)
            logger.info(f"[{symbol}] Loaded cache {interval} {start_date}–{end_date}")
            return df
        except Exception:
            logger.warning(f"[{symbol}] Cache read failed, refetching")
    params = {
        'exchange': 'NSE',
        'symboltoken': token,
        'interval': interval,
        'fromdate': start_date.strftime('%Y-%m-%d 09:15'),
        'todate':   end_date.strftime('%Y-%m-%d 15:30')
    }
    for attempt in range(max_attempts):
        try:
            resp = smart.getCandleData(params)
            data = resp.get('data')
            if data:
                df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.to_pickle(cache_file)
                logger.info(f"[{symbol}] Fetched & cached {len(df)} rows {interval}")
                return df
            else:
                logger.warning(f"[{symbol}] No data for {interval}")
                break
        except smartExceptions.DataException as e:
            logger.warning(f"[{symbol}] DataException on {interval}: {e}")
        except Exception as e:
            logger.error(f"[{symbol}] Fetch error on attempt {attempt+1}: {e}")
        time.sleep(1 * 2**attempt)
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
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def find_pivots(high, low, window=5):
    hi_idx = argrelextrema(high.values, np.greater, order=window)[0]
    return high.iloc[hi_idx].values

def cluster_levels(levels):
    levels = sorted(levels)
    clusters = []
    for lvl in levels:
        if not clusters or abs(lvl - clusters[-1][0])/lvl > CLUSTER_TOL:
            clusters.append([lvl, 1])
        else:
            clusters[-1][0] = (clusters[-1][0]*clusters[-1][1] + lvl) / (clusters[-1][1]+1)
            clusters[-1][1] += 1
    return [c[0] for c in clusters]

def calculate_level_strength(df, level, tol=0.015):
    touches, dates = 0, []
    for idx, row in df.iterrows():
        if abs(row['high'] - level)/level <= tol or abs(row['low'] - level)/level <= tol:
            touches += 1
            dates.append(idx)
    if touches < MIN_TOUCHES:
        return 0, None
    age = (df.index.max().date() - min(dates).date()).days
    if age < MIN_LEVEL_AGE:
        return 0, age
    score = min((touches/5)*0.6 + min(age/90,1)*0.4, 1.0)
    return score, age

def find_strong_resistances(df):
    recent = df if len(df) < VOL_AVG_WINDOW*6 else df.iloc[-VOL_AVG_WINDOW*6:]
    piv_res = find_pivots(recent['high'], recent['low'], window=PIVOT_WINDOW_HOURLY)
    levels = []
    current = df['close'].iloc[-1]
    for raw in cluster_levels(piv_res):
        score, age = calculate_level_strength(df, raw)
        if score >= 0.4 and raw > current:
            levels.append({'level': raw, 'score': score, 'age': age})
    levels.sort(key=lambda x: x['level'])
    return levels

def round_price(price):
    if price < 100:
        return round(price, 2)
    if price < 500:
        return round(price, 1)
    return round(price)

def confirm_breakout(df_day, index, lvl):
    loc = df_day.index.get_loc(index)
    for j in range(CONFIRMATION_CANDLES):
        try:
            price = df_day['close'].iloc[loc + j]
        except IndexError:
            return False
        if price <= lvl:
            return False
    return True

def calculate_final_targets(buy_price, breakout_low, df_daily):
    """Calculate SL and targets using the new logic"""
    # Calculate ATR (14-period)
    high_low = df_daily['high'] - df_daily['low']
    high_close = np.abs(df_daily['high'] - df_daily['close'].shift())
    low_close = np.abs(df_daily['low'] - df_daily['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]
    
    # 1. SL Calculation (Priority-based)
    sl_candidates = []
    
    # Option 1: Breakout low difference (2.5-3.5% range)
    breakout_diff_pct = (buy_price - breakout_low) / buy_price * 100
    if 2.5 <= breakout_diff_pct <= 3.5:
        sl_candidates.append(breakout_low)
    
    # Option 2: ATR method (1.5x ATR in 2.5-3.5% range)
    atr_sl = buy_price - (1.5 * atr)
    atr_diff_pct = (buy_price - atr_sl) / buy_price * 100
    if 2.5 <= atr_diff_pct <= 3.5:
        sl_candidates.append(atr_sl)
    
    # Final SL selection
    if sl_candidates:
        sl = sl_candidates[0]  # Prefer breakout low if both qualify
    else:
        sl = buy_price * 0.97  # Default 3% SL
    
    # Ensure SL is within 2.5-3.5% range as final check
    final_diff_pct = (buy_price - sl) / buy_price * 100
    if final_diff_pct < 2.5:
        sl = buy_price * 0.975  # Minimum 2.5%
    elif final_diff_pct > 3.5:
        sl = buy_price * 0.965  # Maximum 3.5%
    
    # 2. Target Calculation (Progressive Method)
    risk_amount = buy_price - sl
    t1 = buy_price + (risk_amount * 2)
    t2 = t1 + ((t1 - buy_price) * 0.7)
    t3 = t2 + ((t2 - t1) * 0.7)
    
    return {
        'sl': round_price(sl),
        't1': round_price(t1),
        't2': round_price(t2),
        't3': round_price(t3),
        'risk_pct': round(final_diff_pct, 2)
    }

def analyze_stock(smart, symbol, token):
    # determine reference date
    ref_date = TEST_DATE if TEST_MODE else datetime.date.today()
    start_hist = ref_date - datetime.timedelta(days=HISTORY_YEARS*365)
    start_intra = ref_date - datetime.timedelta(days=ANALYSIS_DAYS)

    df_hist = fetch_candles(smart, symbol, token, start_hist, ref_date, HOURLY_INTERVAL)
    if df_hist.empty:
        df_hist = fetch_candles(smart, symbol, token, start_hist, ref_date, DAILY_INTERVAL)
    if df_hist.empty or len(df_hist) < 50:
        return []

    df_intra = fetch_candles(smart, symbol, token, start_intra, ref_date, FIVEMIN_INTERVAL)
    if df_intra.empty:
        df_intra = fetch_candles(smart, symbol, token, start_intra, ref_date, TENMIN_INTERVAL)
    if df_intra.empty:
        return []

    df_intra['rsi'] = compute_rsi(df_intra['close'])
    df_intra['macd'], df_intra['signal'] = compute_macd(df_intra['close'])
    df_intra['vol_avg'] = df_intra['volume'].rolling(VOL_AVG_WINDOW).mean()

    results = []
    for i in range(1, ANALYSIS_DAYS+1):
        day = ref_date - datetime.timedelta(days=i)
        piv_df = df_hist[df_hist.index.date < day]
        if len(piv_df) < 50:
            continue
        levels = find_strong_resistances(piv_df)
        if not levels:
            continue

        day_df = df_intra[df_intra.index.date == day]
        if day_df.empty:
            continue

        candidates = []
        for ts, row in day_df.iterrows():
            if np.isnan(row['vol_avg']) or np.isnan(row['rsi']):
                continue
            loc = day_df.index.get_loc(ts)
            vols = day_df['volume'].iloc[max(0,loc-1):loc+1]
            if not all(vols >= VOL_MULTIPLIER * row['vol_avg']):
                continue

            res_lvls = [l['level'] for l in levels if l['level'] < row['close']]
            if not res_lvls:
                continue
            if not (RSI_RANGE['bullish'][0] <= row['rsi'] <= RSI_RANGE['bullish'][1]):
                continue
            if row['macd'] <= row['signal']:
                continue

            lvl = max(res_lvls)
            if not confirm_breakout(day_df, ts, lvl):
                continue

            # determine buy price as open of the next candle after confirmation
            buy_loc = loc + CONFIRMATION_CANDLES
            if buy_loc >= len(day_df):
                continue
            raw_buy = day_df['open'].iloc[buy_loc]
            buyprice = round_price(raw_buy)

            # With this new logic:
            # Get daily data for ATR calculation
            df_daily = fetch_candles(smart, symbol, token,
                         start_date=day - datetime.timedelta(days=30),
                         end_date=day,
                         interval=DAILY_INTERVAL)
            if df_daily.empty:
                # Fallback to hourly and resample to daily
                df_hour = fetch_candles(smart, symbol, token,
                                        start_date=day - datetime.timedelta(days=30),
                                        end_date=day,
                                        interval=HOURLY_INTERVAL)
                if not df_hour.empty:
                    df_daily = df_hour.resample('1D').agg({
                        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
                    }).dropna()
            if df_daily.empty:
                continue
                
            targets = calculate_final_targets(
                buy_price=buyprice,
                breakout_low=lvl,
                df_daily=df_daily
            )
            
            if not (2.5 <= targets['risk_pct'] <= 3.5):
                continue  # Skip trades with abnormal risk
            
            sl, t1, t2, t3 = targets['sl'], targets['t1'], targets['t2'], targets['t3']
            candidates.append({
                'date': day, 'time': ts.time(),
                'symbol': symbol, 'token': token,
                'action': 'BUY', 'breakout_level': round_price(lvl),
                'buyprice': buyprice, 'stop_loss': sl,
                'target1': t1, 'target2': t2, 'target3': t3,
                'rsi': row['rsi'], 'macd': row['macd'], 'volume': row['volume'],
                'close': row['close']
            })
        if candidates:
            best = max(candidates, key=lambda x: x['volume'])
            results.append(best)
    return results

def main():
    smart = init_smartapi()
    full_map = load_symbol_tokens(SYMBOL_TOKEN_FILE)
    all_results = []
    for i, (sym, tok) in enumerate(full_map.items(),1):
        if i>201: break
        logger.info(f"Processing {sym} ({i}/{len(full_map)})")
        all_results.extend(analyze_stock(smart, sym, tok))
        time.sleep(0.2)

    df = pd.DataFrame(all_results)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['symbol','date','time'], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == '__main__':
    main()
