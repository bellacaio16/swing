import os
import datetime
import pandas as pd
import numpy as np
import time
import logging
from scipy.signal import argrelextrema
import requests

# ======= CONFIGURATION ========
SYMBOLS_CSV      = 'symbols_data.csv'
HISTORY_YEARS    = 2
ANALYSIS_DAYS    = 60
OUTPUT_CSV       = 'breakouts.csv'
CACHE_DIR        = 'cache'
HOURLY_INTERVAL  = 'ONE_HOUR'
FIVEMIN_INTERVAL = 'FIVE_MINUTE'
TENMIN_INTERVAL  = 'TEN_MINUTE'

TEST_MODE = True
TEST_DATE = datetime.date(2025, 6, 20)

MIN_LEVEL_AGE  = 30
MIN_TOUCHES    = 3
VOL_MULTIPLIER = 1.5
RSI_RANGE      = {'bullish': (60, 70)}
CONFIRM_CANDLES= 2

RSI_PERIOD     = 14
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
VOL_AVG_WINDOW = 20
PIVOT_WINDOW   = 24
CLUSTER_TOL    = 0.01

MARKET_START_TIME = datetime.time(9, 15)
MARKET_END_TIME   = datetime.time(15, 30)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()

def load_symbols_data(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return {
        row['tradingsymbol']: {
            'instrument_key': row['instrument_key'],
            'token': row['exchange_token']
        }
        for _, row in df.iterrows()
    }

def fetch_candles_upstox(instrument_key, start_date, end_date, interval):
    """Chunked + cached fetch, but never cache empties."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    sd = start_date.date() if isinstance(start_date, datetime.datetime) else start_date
    ed = end_date.date()   if isinstance(end_date, datetime.datetime)   else end_date

    interval_map = {
        'FIVE_MINUTE': ('minutes', 5,  30),
        'TEN_MINUTE':  ('minutes',10,  30),
        'ONE_HOUR':    ('hours',   1,  90),
    }
    if interval not in interval_map:
        raise ValueError(f"Unsupported interval: {interval}")
    unit, count, max_days = interval_map[interval]

    key_safe = instrument_key.replace('|','_')
    cache_file = os.path.join(
        CACHE_DIR,
        f"{key_safe}_{interval}_{sd.strftime('%Y%m%d')}_{ed.strftime('%Y%m%d')}.pkl"
    )
    # load cache if nonempty
    if os.path.exists(cache_file):
        mod = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.datetime.now() - mod).days < 7:
            df = pd.read_pickle(cache_file)
            if not df.empty:
                logger.info(f"Using cache for {instrument_key} [{sd}→{ed}]")
                return df

    # build chunks
    chunks = []
    cur = sd
    while cur <= ed:
        end_chunk = min(ed, cur + datetime.timedelta(days=max_days-1))
        chunks.append((cur, end_chunk))
        cur = end_chunk + datetime.timedelta(days=1)

    frames = []
    for cs, ce in chunks:
        cs_str, ce_str = cs.strftime('%Y-%m-%d'), ce.strftime('%Y-%m-%d')
        url = (
            f"https://api.upstox.com/v3/historical-candle/"
            f"{instrument_key}/{unit}/{count}/{ce_str}/{cs_str}"
        )
        logger.info(f"Fetching {instrument_key} {interval} [{cs_str}→{ce_str}]")
        for attempt in range(3):
            try:
                resp = requests.get(url, headers={'Accept':'application/json'})
                resp.raise_for_status()
                data = resp.json().get('data',{})
                c = data.get('candles', [])
                if not c:
                    logger.debug(" no candles returned")
                    break
                df_tmp = pd.DataFrame(c, columns=[
                    'timestamp','open','high','low','close','volume','_'
                ]).drop(columns=['_'])
                df_tmp['timestamp'] = pd.to_datetime(df_tmp['timestamp'])
                df_tmp.set_index('timestamp', inplace=True)
                df_tmp.index = df_tmp.index.tz_localize(None)
                df_tmp = df_tmp.between_time(MARKET_START_TIME, MARKET_END_TIME)
                if not df_tmp.empty:
                    frames.append(df_tmp)
                time.sleep(0.5)
                break
            except Exception as e:
                logger.warning(f" attempt {attempt+1} failed: {e}")
                time.sleep(2**attempt)

    if not frames:
        logger.warning(f"No data for {instrument_key} [{sd}→{ed}]")
        return pd.DataFrame()

    df_all = pd.concat(frames).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep='first')]
    df_all.to_pickle(cache_file)
    logger.info(f"Cached {len(df_all)} rows for {instrument_key}")
    return df_all

def compute_rsi(s, period=RSI_PERIOD):
    d = s.diff().dropna()
    g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al
    return 100 - (100/(1+rs))

def compute_macd(s, fast=MACD_FAST, slow=MACD_SLOW, sig=MACD_SIGNAL):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    m = ef - es
    si= m.ewm(span=sig, adjust=False).mean()
    return m, si

def find_pivots(high, low, window=PIVOT_WINDOW):
    idx = argrelextrema(high.values, np.greater, order=window)[0]
    return high.iloc[idx].values

def cluster_levels(levels):
    levels = sorted(levels)
    cl = []
    for lv in levels:
        if not cl or abs(lv-cl[-1][0])/lv > CLUSTER_TOL:
            cl.append([lv,1])
        else:
            cl[-1][0] = (cl[-1][0]*cl[-1][1]+lv)/(cl[-1][1]+1)
            cl[-1][1] += 1
    return [c[0] for c in cl]

def calculate_level_strength(df, lvl, tol=0.015):
    touches, dates = 0, []
    for i,row in df.iterrows():
        if abs(row['high']-lvl)/lvl<=tol or abs(row['low']-lvl)/lvl<=tol:
            touches+=1; dates.append(i)
    if touches<MIN_TOUCHES:
        return 0, None
    age = (df.index.max().date()-min(dates).date()).days
    if age<MIN_LEVEL_AGE:
        return 0, age
    score = min((touches/5)*0.6 + min(age/90,1)*0.4, 1)
    return score, age

def find_strong_resistances(df):
    sample = df if len(df)<VOL_AVG_WINDOW*6 else df.iloc[-VOL_AVG_WINDOW*6:]
    pivs   = find_pivots(sample['high'], sample['low'])
    levels = []
    cur    = df['close'].iloc[-1]
    for raw in cluster_levels(pivs):
        score,age = calculate_level_strength(df,raw)
        if score>=0.4 and raw>cur:
            levels.append({'level':raw,'score':score,'age':age})
    return sorted(levels, key=lambda x:x['level'])

def round_price(p):
    if p<100: return round(p,2)
    if p<500: return round(p,1)
    return round(p)

def confirm_breakout(df, idx, lvl):
    try:
        loc = df.index.get_loc(idx)
        for j in range(CONFIRM_CANDLES):
            if df['close'].iloc[loc+j] <= lvl:
                return False
        return True
    except (KeyError, IndexError):
        return False

def calculate_final_targets(buy, low, df_d):
    hl = df_d['high']-df_d['low']
    hc = abs(df_d['high']-df_d['close'].shift())
    lc = abs(df_d['low'] -df_d['close'].shift())
    tr = pd.concat([hl,hc,lc],axis=1).max(axis=1)
    atr= tr.rolling(14).mean().iloc[-1]

    cands=[]
    pct = (buy-low)/buy*100
    if 2.5<=pct<=3.5: cands.append(low)
    atr_sl=buy-1.5*atr
    if 2.5<=((buy-atr_sl)/buy*100)<=3.5: cands.append(atr_sl)

    sl = cands[0] if cands else buy*0.97
    final_pct=(buy-sl)/buy*100
    if final_pct<2.5: sl=buy*0.975
    if final_pct>3.5: sl=buy*0.965

    risk = buy-sl
    t1 = buy+2*risk
    t2 = t1+0.7*(t1-buy)
    t3 = t2+0.7*(t2-t1)

    return {'sl':round_price(sl),'t1':round_price(t1),
            't2':round_price(t2),'t3':round_price(t3),
            'risk_pct':round((buy-sl)/buy*100,2)}

def analyze_stock(sym,data):
    ref = TEST_DATE if TEST_MODE else datetime.date.today()
    hstart = ref - datetime.timedelta(days=HISTORY_YEARS*365)
    istart = ref - datetime.timedelta(days=ANALYSIS_DAYS)

    ik = data['instrument_key']
    # hourly history
    df_h = fetch_candles_upstox(ik,hstart,ref,HOURLY_INTERVAL)
    if df_h.empty or len(df_h)<50:
        return []

    # intraday
    df_i = fetch_candles_upstox(ik,istart,ref,FIVEMIN_INTERVAL)
    if df_i.empty:
        df_i = fetch_candles_upstox(ik,istart,ref,TENMIN_INTERVAL)
    if df_i.empty:
        return []

    # indicators
    df_i['rsi'], df_i['macd'] = compute_rsi(df_i['close']), None
    df_i['macd'], df_i['signal'] = compute_macd(df_i['close'])
    df_i['vol_avg'] = df_i['volume'].rolling(VOL_AVG_WINDOW).mean()

    res=[]
    for d in range(1, ANALYSIS_DAYS+1):
        day = ref - datetime.timedelta(days=d)
        piv_df = df_h[df_h.index.date < day]
        if len(piv_df)<50: continue

        levels = find_strong_resistances(piv_df)
        if not levels: continue

        day_df = df_i[df_i.index.date==day]
        if day_df.empty: continue

        cands=[]
        for ts,row in day_df.iterrows():
            if np.isnan(row['vol_avg']) or np.isnan(row['rsi']):
                continue
            loc = day_df.index.get_loc(ts)
            vols=day_df['volume'].iloc[max(0,loc-1):loc+1]
            if not all(vols>=VOL_MULTIPLIER*row['vol_avg']):
                continue
            prev_levels=[l['level'] for l in levels if l['level']<row['close']]
            if not prev_levels: continue
            if not (RSI_RANGE['bullish'][0] <= row['rsi'] <= RSI_RANGE['bullish'][1]):
                continue
            if row['macd']<=row['signal']:
                continue
            lvl=max(prev_levels)
            if not confirm_breakout(day_df,ts,lvl):
                continue

            buying_idx=loc+CONFIRM_CANDLES
            if buying_idx>=len(day_df):
                continue
            buy_price=round_price(day_df['open'].iloc[buying_idx])

            # build daily from hourly
            hourly_30 = fetch_candles_upstox(ik,day - datetime.timedelta(days=30), day, HOURLY_INTERVAL)
            if hourly_30.empty:
                continue
            df_daily = hourly_30.resample('1D').agg({
                'open':'first','high':'max','low':'min','close':'last','volume':'sum'
            }).dropna()

            targets = calculate_final_targets(buy_price, lvl, df_daily)
            if not (2.5<=targets['risk_pct']<=3.5):
                continue

            cands.append({
                'date':day,'time':ts.time(),'symbol':sym,'token':data['token'],
                'action':'BUY','breakout_level':round_price(lvl),
                'buyprice':buy_price,'stop_loss':targets['sl'],
                'target1':targets['t1'],'target2':targets['t2'],'target3':targets['t3'],
                'rsi':row['rsi'],'macd':row['macd'],
                'volume':row['volume'],'close':row['close']
            })
        if cands:
            res.append(max(cands, key=lambda x:x['volume']))
    return res

def main():
    symbols = load_symbols_data(SYMBOLS_CSV)
    all_results=[]
    for i,(sym,data) in enumerate(symbols.items(),1):
        if i>200: break
        logger.info(f"Processing {sym} ({i}/{len(symbols)})")
        try:
            picks = analyze_stock(sym,data)
            all_results.extend(picks)
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"Error with {sym}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        df.sort_values(['symbol','date','time'], inplace=True)
        df.to_csv(OUTPUT_CSV, index=False)
    else:
        logger.warning("No breakout candidates found")

if __name__=='__main__':
    main()
