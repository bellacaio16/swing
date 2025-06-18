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
MAX_HOLD_DAYS = 7               # Max trading days to hold
FORCE_EXIT_DAYS = 3             # Force exit if no target hit within this many days
RISK_PER_TRADE_PCT = 0.035      # 3.5% of capital per trade
MAX_CAPITAL_PER_TRADE = 35000   # Hard cap per trade
ATR_PERIOD = 14                 # ATR lookback period
ATR_MULTIPLIER = 1.5            # For trailing stop calculation
CACHE_MAX_AGE_DAYS = 7          # Invalidate cached files older than this
TOTAL_CAPITAL_START = 200000    # ₹2,00,000 initial capital
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

# Calculate total charges for CNC delivery trades
def calculate_charges(buy_val, sell_val):
    stamp = 0.00015 * buy_val            # Stamp Duty on Buy (0.015%)
    stt = 0.001 * sell_val               # STT on Sell (0.1%)
    exch = 0.0000345 * (buy_val + sell_val)  # Exchange Transaction Charges (0.00345%)
    sebi = 0.000001 * (buy_val + sell_val)   # SEBI Charges (0.0001%)
    gst = 0.18 * (exch + sebi)           # GST (18% on exchange + SEBI)
    dp = 15.93                           # DP Charges fixed per sell
    return stamp + stt + exch + sebi + gst + dp


def fetch_daily_candles(smart, token, start_date, end_date):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(
        CACHE_DIR,
        f"{token}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
    )

    if os.path.exists(cache_file):
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.datetime.now() - mod_time).days < CACHE_MAX_AGE_DAYS:
            df = pd.read_pickle(cache_file)
            print("Cache loaded")
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            return df

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
            df = pd.concat(parts)
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            df = df.sort_index()
            df.to_pickle(cache_file)
            return df
        return pd.DataFrame()

    params = {
        'exchange': 'NSE',
        'symboltoken': token,
        'interval': DAILY_INTERVAL,
        'fromdate': start_date.strftime('%Y-%m-%d 00:00'),
        'todate': end_date.strftime('%Y-%m-%d 23:59')
    }
    for attempt in range(5):
        try:
            resp = smart.getCandleData(params)
            data = resp.get('data') or []
            print("Fetched")
            break
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Fetch error for {token} (attempt {attempt+1}), retrying in {wait}s: {e}")
            time.sleep(wait)
    else:
        logger.error(f"Failed to fetch data for {token}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize(None)
    df.to_pickle(cache_file)
    time.sleep(0.2)
    return df


def compute_atr(df, period=ATR_PERIOD):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def main():
    smart = init_smartapi()
    raw = pd.read_csv(BREAKOUT_CSV, dtype=str)
    raw['time'] = raw.get('time', '09:15:00')
    raw['date'] = pd.to_datetime(raw['date'], dayfirst=True).dt.date
    raw['datetime'] = pd.to_datetime(raw['date'].astype(str) + ' ' + raw['time'])
    raw = raw.sort_values('datetime')

    start_date = raw['date'].min() - timedelta(days=ATR_PERIOD+5)
    last_signal = raw['date'].max() + timedelta(days=MAX_HOLD_DAYS)

    history = {}
    for tok in raw['token'].unique():
        df = fetch_daily_candles(smart, tok, start_date, last_signal)
        if df.empty:
            continue
        df['ATR'] = compute_atr(df)
        history[tok] = df

    trades_by_date = raw.groupby('date')
    total_capital = TOTAL_CAPITAL_START
    ongoing = {}
    results = []
    current = raw['date'].min()

    while current <= last_signal:
        # EXIT logic
        for sym, pos in list(ongoing.items()):
            df_hist = history.get(pos['token'])
            if df_hist is None or current not in df_hist.index.date:
                continue
            row = df_hist.loc[df_hist.index.date == current].iloc[0]
            high, low, close, atr = row['high'], row['low'], row['close'], row['ATR']
            pos['days_held'] += 1
            pos['highest'] = max(pos.get('highest', pos['breakout']), high)
            pos['trail_stop'] = pos['highest'] - ATR_MULTIPLIER * atr

            exit_flag = False
            if low <= pos['sl']:
                price, reason = pos['sl'], 'SL'; exit_flag = True
            elif pos['days_held'] <= FORCE_EXIT_DAYS and low <= pos['breakout']:
                price, reason = pos['breakout'], 'QUALITY_EXIT'; exit_flag = True
            elif pos['hit1'] and not pos['hit2'] and low <= pos['breakout']:
                price, reason = pos['breakout'], 'T1_SL'; exit_flag = True
            elif pos['hit2'] and low <= pos['targets'][0]:
                price, reason = pos['targets'][0], 'T2_SL'; exit_flag = True
            elif low <= pos['trail_stop']:
                price, reason = pos['trail_stop'], 'ATR_TRAIL'; exit_flag = True
            elif high >= pos['targets'][2]:
                price, reason = pos['targets'][2], 'T3'; exit_flag = True
            elif pos['days_held'] >= FORCE_EXIT_DAYS and not (pos['hit1'] or pos['hit2']):
                price, reason = close, 'FORCE_EXIT'; exit_flag = True
            elif pos['days_held'] > MAX_HOLD_DAYS:
                price, reason = close, 'TIME_STOP'; exit_flag = True

            if exit_flag:
                buy_val = pos['breakout'] * pos['size']
                sell_val = price * pos['size']
                charges = calculate_charges(buy_val, sell_val)
                pnl = (price - pos['breakout']) * pos['size']
                total_capital += pos['capital'] + pnl
                print("EXIT--")
                results.append({**pos,
                                'exit_date': current.strftime('%Y-%m-%d'),
                                'exit_price': price,
                                'result': reason,
                                'pnl_total': pnl,
                                'charges': round(charges,2),
                                'remaining_capital': total_capital})
                del ongoing[sym]

        # ENTRY logic
        if current in trades_by_date.groups:
            for _, row in trades_by_date.get_group(current).iterrows():
                sym, tok = row['symbol'], row['token']
                breakout = float(row['breakout_level'])
                df_hist = history.get(tok)
                if df_hist is None or current not in df_hist.index.date:
                    continue
                atr = df_hist.loc[df_hist.index.date == current, 'ATR'].iloc[0]
                if np.isnan(atr) or atr <= 0:
                    continue
                risk_amount = min(RISK_PER_TRADE_PCT * total_capital, MAX_CAPITAL_PER_TRADE)
                size = int(np.floor(risk_amount / (atr * ATR_MULTIPLIER)))
                if size < 1:
                    continue
                cap_used = size * breakout
                if cap_used > MAX_CAPITAL_PER_TRADE:
                    size = MAX_CAPITAL_PER_TRADE // breakout
                    cap_used = size * breakout

                if sym in ongoing:
                    old = ongoing.pop(sym)
                    open_price = df_hist.loc[df_hist.index.date == current, 'open'].iloc[0]
                    buy_val_old = old['breakout'] * old['size']
                    sell_val_old = open_price * old['size']
                    charges_old = 0
                    pnl = (open_price - old['breakout']) * old['size']
                    total_capital += old['capital'] + pnl
                    results.append({**old,
                                    'exit_date': current.strftime('%Y-%m-%d'),
                                    'exit_price': open_price,
                                    'result': 'REPLACE',
                                    'pnl_total': pnl,
                                    'charges': round(charges_old,2),
                                    'remaining_capital': total_capital})
                if total_capital >= cap_used:
                    total_capital -= cap_used
                    print("-----BUY")
                    ongoing[sym] = {
                        'symbol': sym,
                        'entry_date': current.strftime('%Y-%m-%d'),
                        'breakout': breakout,
                        'sl': float(row['stop_loss']),
                        'targets': (
                            float(row['target1']),
                            float(row['target2']),
                            float(row['target3'])
                        ),
                        'token': tok,
                        'size': size,
                        'capital': cap_used,
                        'hit1': False,
                        'hit2': False,
                        'days_held': 0
                    }
        current += timedelta(days=1)

    # Final exit of open positions
    last_date = last_signal
    for sym, pos in ongoing.items():
        df_hist = history.get(pos['token'], pd.DataFrame())
        if not df_hist.empty and last_date in df_hist.index.date:
            close_price = df_hist.loc[df_hist.index.date == last_date, 'close'].iloc[0]
        else:
            close_price = pos['breakout']
        buy_val = pos['breakout'] * pos['size']
        sell_val = close_price * pos['size']
        charges = calculate_charges(buy_val, sell_val)
        pnl = (close_price - pos['breakout']) * pos['size']
        total_capital += pos['capital'] + pnl
        results.append({**pos,
                        'exit_date': last_date.strftime('%Y-%m-%d'),
                        'exit_price': close_price,
                        'result': 'END',
                        'pnl_total': pnl,
                        'charges': round(charges,2),
                        'remaining_capital': total_capital})
        print("EXIT--")

        if results:
            df_results = pd.DataFrame(results)

            # Calculate total pnl and total charges
            total_pnl = df_results['pnl_total'].sum()
            total_charges = df_results['charges'].sum()

            # Append a summary row at the end
            summary_row = {col: '' for col in df_results.columns}
            summary_row['result'] = 'TOTAL'
            summary_row['pnl_total'] = total_pnl
            summary_row['charges'] = round(total_charges, 2)
            df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)

            # Save to CSV
            df_results.to_csv(SWING_BACKTEST_CSV, index=False)
            print(f"Backtest complete, results saved to {SWING_BACKTEST_CSV}")
            print(f"\n==== SUMMARY ====")
            print(f"Total PnL     : ₹{round(total_pnl, 2)}")
            print(f"Total Charges : ₹{round(total_charges, 2)}\n")
            print(f"Net Profit : ₹{round(total_pnl, 2) - round(total_charges, 2)}\n")
        else:
            logger.warning("No backtest results to save")


if __name__ == '__main__':
    main()
