import os
import datetime
import pandas as pd
import numpy as np
from datetime import timedelta
import time
import logging
import requests
from pandas.tseries.offsets import BDay

# ======= CONFIGURATION ========
BREAKOUT_CSV = 'breakouts.csv'
SWING_BACKTEST_CSV = 'backtest_results.csv'
SYMBOLS_CSV = 'symbols_data.csv'
CACHE_DIR = 'cache_10min'
INTRADAY_UNIT = 'minutes'
INTRADAY_INTERVAL = 10
MAX_HOLD_DAYS = 8
FORCE_EXIT_DAYS = 6
MAX_CAPITAL_PER_TRADE = 35000
RISK_PER_TRADE_PCT = 0.030
ATR_PERIOD = 14
CACHE_MAX_AGE_DAYS = 7
TOTAL_CAPITAL_START = 200000
MARKET_START_TIME = datetime.time(9, 15)
MARKET_END_TIME = datetime.time(15, 30)
MAX_DATA_STALENESS_MINUTES = 15
# ==============================

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Market holidays (add dates as needed)
MARKET_HOLIDAYS = [
    datetime.date(2023, 1, 26),
    datetime.date(2023, 3, 7),
    datetime.date(2023, 3, 30),
    datetime.date(2023, 4, 4),
    datetime.date(2023, 4, 7),
    datetime.date(2023, 4, 14),
    datetime.date(2023, 5, 1),
    datetime.date(2023, 6, 28),
    datetime.date(2023, 8, 15),
    datetime.date(2023, 9, 19),
    datetime.date(2023, 10, 2),
    datetime.date(2023, 10, 24),
    datetime.date(2023, 11, 14),
    datetime.date(2023, 11, 27),
    datetime.date(2023, 12, 25)
]

# Replace the MARKET_HOLIDAYS list and is_market_open function with this:

def is_market_open(date, instrument_key=None, cached_data=None):
    """Check if market is open on given date"""
    # Convert to date object if it's datetime
    if isinstance(date, datetime.datetime):
        date = date.date()
    
    # Always closed on weekends
    if date.weekday() >= 5:
        return False
        
    # Check manual holiday list (optional)
    MARKET_HOLIDAYS = []  # Empty by default, add dates if needed
    
    if date in MARKET_HOLIDAYS:
        return False
        
    # If instrument data is provided, verify we have data for this date
    if instrument_key and cached_data:
        date_str = date.strftime('%Y-%m-%d')
        df_day = cached_data[cached_data.index.date == date]
        if df_day.empty:
            logger.info(f"No data available for {instrument_key} on {date_str} - treating as holiday")
            return False
            
    return True

def calculate_charges(buy_val, sell_val):
    stamp = 0.00015 * buy_val
    stt = 0.001 * (sell_val + buy_val)
    exch = 0.0000345 * (buy_val + sell_val)
    sebi = 0.000001 * (buy_val + sell_val)
    gst = 0.18 * (exch + sebi)
    dp = 14.75
    return stamp + stt + dp + gst + exch + sebi

def fetch_intraday_candles(instrument_key, start_date, end_date):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key_safe = instrument_key.replace('|', '_')
    cache_file = os.path.join(
        CACHE_DIR,
        f"{key_safe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
    )
    
    # Check cache first
    if os.path.exists(cache_file):
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.datetime.now() - mod_time).days < CACHE_MAX_AGE_DAYS:
            df = pd.read_pickle(cache_file)
            if not df.empty:
                logger.info(f"Using cached data for {instrument_key}")
                return df

    # The Upstox API returns data in {'candles': [...]} format
    all_data = []
    current_date = start_date.date()
    end_date = end_date.date()
    
    while current_date <= end_date:
        if not is_market_open(current_date):
            current_date += timedelta(days=1)
            continue
            
        date_str = current_date.strftime('%Y-%m-%d')
        url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/{INTRADAY_INTERVAL}/{date_str}/{date_str}"
        
        logger.info(f"Fetching {instrument_key} for {date_str}")
        
        for attempt in range(3):
            try:
                resp = requests.get(url, headers={'Accept': 'application/json'})
                resp.raise_for_status()
                data = resp.json()
                
                if not data or 'data' not in data or 'candles' not in data['data']:
                    logger.debug(f"No candle data for {instrument_key} on {date_str}")
                    break
                
                candles = data['data']['candles']
                df_day = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'other'])
                df_day = df_day.drop(columns=['other'])
                df_day['timestamp'] = pd.to_datetime(df_day['timestamp'])
                df_day.set_index('timestamp', inplace=True)
                df_day.index = df_day.index.tz_localize(None)
                df_day = df_day.between_time(MARKET_START_TIME, MARKET_END_TIME)
                
                if not df_day.empty:
                    all_data.append(df_day)
                
                time.sleep(0.5)  # Rate limiting
                break
                
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"Attempt {attempt+1} failed for {date_str}: {str(e)}")
                if attempt == 2:
                    logger.error(f"Failed to fetch {instrument_key} for {date_str}")
                time.sleep(wait)
        
        current_date += timedelta(days=1)

    if not all_data:
        logger.warning(f"No data found for {instrument_key} between {start_date} and {end_date}")
        return pd.DataFrame()

    df = pd.concat(all_data).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.to_pickle(cache_file)
    logger.info(f"Cached {len(df)} candles for {instrument_key}")
    return df

def compute_atr(df, period=ATR_PERIOD):
    if len(df) < period:
        return pd.Series(np.nan, index=df.index)
        
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def generate_trading_timestamps(start_date, end_date):
    """Generate 10-minute timestamps for trading days between start_date and end_date"""
    timestamps = []
    current_date = start_date
    
    while current_date <= end_date:
        if not is_market_open(current_date):
            current_date += timedelta(days=1)
            continue
            
        current_time = datetime.datetime.combine(current_date, MARKET_START_TIME)
        end_time = datetime.datetime.combine(current_date, MARKET_END_TIME)
        
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += timedelta(minutes=INTRADAY_INTERVAL)
            
        current_date += timedelta(days=1)
        
    return timestamps

def main():
    # Load and preprocess data
    raw = pd.read_csv(BREAKOUT_CSV)
    raw.columns = [c.lower() for c in raw.columns]
    
    # Remove -EQ suffix from symbols
    raw['symbol'] = raw['symbol'].str.replace('-EQ', '').str.strip()
    
    # Handle datetime columns
    if 'datetime' in raw.columns:
        raw['datetime'] = pd.to_datetime(raw['datetime'])
    elif 'date' in raw.columns and 'time' in raw.columns:
        raw['datetime'] = pd.to_datetime(raw['date'] + ' ' + raw['time'])
    elif 'date' in raw.columns:
        raw['datetime'] = pd.to_datetime(raw['date']) + pd.Timedelta(hours=9, minutes=15)
    else:
        raise ValueError("CSV must contain either 'datetime' column or 'date' column")
    
    raw['token'] = raw['token'].astype(int)
    raw = raw.sort_values('datetime')

    # Merge with symbols data
    symbols = pd.read_csv(SYMBOLS_CSV)
    symbols.columns = [c.lower() for c in symbols.columns]
    symbols['exchange_token'] = symbols['exchange_token'].astype(int)
    raw = raw.merge(
        symbols[['instrument_key', 'exchange_token']],
        left_on='token', 
        right_on='exchange_token', 
        how='left'
    )

    # Prepare timeline
    first_signal = raw['datetime'].min().to_pydatetime()
    last_signal = raw['datetime'].max().to_pydatetime()
    end_date = last_signal + timedelta(days=MAX_HOLD_DAYS + 1)
    timestamps = generate_trading_timestamps(first_signal.date(), end_date.date())
    
    # Initialize trading variables
    total_capital = TOTAL_CAPITAL_START
    ongoing_positions = {}
    results = []
    cached_data = {}
    
    logger.info(f"Starting backtest from {first_signal} to {end_date}")
    logger.info(f"Total timestamps to process: {len(timestamps)}")
    logger.info(f"Initial capital: {total_capital}")

    # Main processing loop
    for idx, current_ts in enumerate(timestamps):
        if idx % 100 == 0:
            logger.info(f"Processing {idx+1}/{len(timestamps)}: {current_ts}")
        
        # EXIT LOGIC
        for symbol, pos in list(ongoing_positions.items()):
            df = cached_data.get(pos['instrument_key'])
            if df is None:
                logger.warning(f"No data cached for {symbol}")
                continue
                
            try:
                current_bar = df.loc[current_ts]
            except KeyError:
                try:
                    current_bar = df.iloc[df.index.get_indexer([current_ts], method='nearest')[0]]
                except:
                    logger.warning(f"No price data for {symbol} at {current_ts}")
                    continue
            
            high = current_bar['high']
            low = current_bar['low']
            close = current_bar['close']
            ongoing_positions[symbol]['current_price'] = close
            
            # Calculate trading days held (excluding weekends/holidays)
            hold_days = len(pd.bdate_range(pos['entry_date'], current_ts.date()))
            
            exit_reason = None
            exit_price = None
            
            # Target 1 hit
            if not pos['hit1'] and high >= pos['targets'][0]:
                ongoing_positions[symbol]['hit1'] = True
                ongoing_positions[symbol]['stop_loss'] = pos['buyprice'] + (pos['buyprice'] - pos['initial_sl'])
                logger.info(f"{symbol} hit T1 at {pos['targets'][0]}, new SL: {ongoing_positions[symbol]['stop_loss']}")
            
            # Target 2 hit
            if pos['hit1'] and not pos['hit2'] and high >= pos['targets'][1]:
                ongoing_positions[symbol]['hit2'] = True
                ongoing_positions[symbol]['stop_loss'] += (pos['targets'][1] - pos['targets'][0])
                logger.info(f"{symbol} hit T2 at {pos['targets'][1]}, new SL: {ongoing_positions[symbol]['stop_loss']}")
            
            # Exit conditions
            if high >= pos['targets'][2]:
                exit_reason = 'T3'
                exit_price = pos['targets'][2]
            elif low <= pos['stop_loss']:
                exit_reason = 'SL'
                if low >= pos['targets'][0]:
                    exit_reason = 'T1_SL'
                if low >= pos['targets'][1]:
                    exit_reason = 'T2_SL'
                exit_price = pos['stop_loss']
            elif hold_days >= FORCE_EXIT_DAYS and not (pos['hit1'] or pos['hit2']):
                exit_reason = 'FORCE_EXIT'
                exit_price = close
            elif hold_days > MAX_HOLD_DAYS:
                exit_reason = 'TIME_STOP'
                exit_price = close
            
            if exit_reason:
                buy_val = pos['buyprice'] * pos['size']
                sell_val = exit_price * pos['size']
                charges = calculate_charges(buy_val, sell_val)
                pnl = (exit_price - pos['buyprice']) * pos['size'] - charges
                
                total_capital += pos['capital'] + pnl
                
                trade_result = {
                    'symbol': symbol,
                    'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'entry_time': pos['entry_time'].strftime('%H:%M'),
                    'exit_date': current_ts.strftime('%Y-%m-%d'),
                    'exit_time': current_ts.strftime('%H:%M'),
                    'buyprice': pos['buyprice'],
                    'exit_price': exit_price,
                    'size': pos['size'],
                    'stop_loss': pos['stop_loss'],
                    'target1': pos['targets'][0],
                    'target2': pos['targets'][1],
                    'target3': pos['targets'][2],
                    'result': exit_reason,
                    'pnl_total': pnl,
                    'charges': round(charges, 2),
                    'remaining_capital': total_capital,
                    'days_held': hold_days
                }
                results.append(trade_result)
                logger.info(f"EXITED {symbol} at {exit_price} ({exit_reason}), P&L: {pnl:.2f}")
                del ongoing_positions[symbol]
        
      # ENTRY LOGIC
        signal_start = current_ts - timedelta(minutes=10)
        signal_end = current_ts  # Current timestamp
        signals = raw[
            (raw['datetime'] > signal_start) & 
            (raw['datetime'] <= signal_end)
        ]
        
        for _, row in signals.iterrows():
            print("hi")
            symbol = row['symbol']
            inst_key = row['instrument_key']
            
            if symbol in ongoing_positions:
                logger.debug(f"Skipping {symbol} - already in portfolio")
                continue
                
            if inst_key not in cached_data:
                logger.info(f"Fetching data for {symbol} ({inst_key})")
                start_fetch = current_ts - timedelta(days=20)
                end_fetch = current_ts + timedelta(days=10)
                df = fetch_intraday_candles(inst_key, start_fetch, end_fetch)
                
                if df.empty:
                    logger.warning(f"No data found for {symbol} ({inst_key})")
                    continue
                    
                print("fetched")
                cached_data[inst_key] = df
            
            df = cached_data[inst_key]
            
            # Verify sufficient future data exists (10 days)
            future_data_points = len(df[df.index >= current_ts])
            if future_data_points < 38:  # ~3.8 candles per day * 10 days
                logger.warning(f"Insufficient future data for {symbol} (only {future_data_points} points)")
                continue
            
            try:
                current_bar = df[df.index >= row['datetime']].iloc[0]
                if (current_bar.name - row['datetime']) > timedelta(minutes=MAX_DATA_STALENESS_MINUTES):
                    logger.warning(f"Price data too stale for {symbol}")
                    continue
                    
                current_price = current_bar['close']
            except IndexError:
                logger.warning(f"No price data available for {symbol} after {row['datetime']}")
                continue
            
            # Calculate position size with proper capital constraints
            atr_series = compute_atr(df.tail(100))
            if atr_series.empty or atr_series.isna().all():
                logger.warning(f"Invalid ATR for {symbol}")
                continue
                
            atr = atr_series.iloc[-1]
            risk_amount = min(RISK_PER_TRADE_PCT * total_capital, MAX_CAPITAL_PER_TRADE)
            
            # Calculate max shares within capital limits
            max_shares_by_risk = max(1, int(risk_amount / (atr * 1.5)))
            max_shares_by_capital = min(MAX_CAPITAL_PER_TRADE // current_price, 
                                      total_capital // current_price)
            
            position_size = min(max_shares_by_risk, max_shares_by_capital)
            if position_size < 1:
                logger.info(f"Position size too small for {symbol} at {current_price}")
                continue
                
            trade_cost = current_price * position_size
            if trade_cost > total_capital:
                logger.info(f"Insufficient capital for {symbol} (needed: {trade_cost:.2f}, available: {total_capital:.2f})")
                continue
                
            # Execute trade
            total_capital -= trade_cost
            ongoing_positions[symbol] = {
                'symbol': symbol,
                'instrument_key': inst_key,
                'entry_date': current_ts.date(),
                'entry_time': current_ts.time(),
                'buyprice': current_price,
                'initial_sl': row['stop_loss'],
                'stop_loss': row['stop_loss'],
                'targets': (row['target1'], row['target2'], row['target3']),
                'size': position_size,
                'capital': trade_cost,
                'hit1': False,
                'hit2': False,
                'current_price': current_price
            }
            
            logger.info(f"ENTERED {symbol} at {current_price:.2f} for {position_size} shares")
            logger.info(f"Remaining capital: {total_capital:.2f}")

    # Final cleanup for positions still open
    for symbol, pos in ongoing_positions.items():
        df = cached_data.get(pos['instrument_key'])
        if df is not None and not df.empty:
            last_price = df['close'].iloc[-1]
        else:
            last_price = pos['buyprice']
            
        hold_days = len(pd.bdate_range(pos['entry_date'], timestamps[-1].date()))
        buy_val = pos['buyprice'] * pos['size']
        sell_val = last_price * pos['size']
        charges = calculate_charges(buy_val, sell_val)
        pnl = (last_price - pos['buyprice']) * pos['size'] - charges
        
        trade_result = {
            'symbol': symbol,
            'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
            'entry_time': pos['entry_time'].strftime('%H:%M'),
            'exit_date': timestamps[-1].strftime('%Y-%m-%d'),
            'exit_time': timestamps[-1].strftime('%H:%M'),
            'buyprice': pos['buyprice'],
            'exit_price': last_price,
            'size': pos['size'],
            'stop_loss': pos['stop_loss'],
            'target1': pos['targets'][0],
            'target2': pos['targets'][1],
            'target3': pos['targets'][2],
            'result': 'END',
            'pnl_total': pnl,
            'charges': round(charges, 2),
            'remaining_capital': total_capital,
            'days_held': hold_days
        }
        results.append(trade_result)
        logger.info(f"CLOSED {symbol} at end, price: {last_price:.2f}, P&L: {pnl:.2f}")
    
    # Save results
# Save results
# Save results
    if results:
        df_res = pd.DataFrame(results)
        
        # Calculate summary statistics
        total_pnl = df_res['pnl_total'].sum()
        total_charges = df_res['charges'].sum()
        net_profit = total_pnl - total_charges
        final_capital = TOTAL_CAPITAL_START + net_profit
        winning_trades = len(df_res[df_res['pnl_total'] > 0])
        win_rate = winning_trades / len(results) * 100
        avg_hold_days = df_res['days_held'].mean()
        
        # Calculate exit type counts
        exit_counts = df_res['result'].value_counts().to_dict()
        exit_types = {
            'SL': exit_counts.get('SL', 0),
            'FORCE_EXIT': exit_counts.get('FORCE_EXIT', 0),
            'TIME_STOP': exit_counts.get('TIME_STOP', 0),
            'T1': exit_counts.get('T1', 0),
            'T2': exit_counts.get('T2', 0),
            'T3': exit_counts.get('T3', 0),
            'T1_SL': exit_counts.get('T1_SL', 0),
            'T2_SL': exit_counts.get('T2_SL', 0),
        }
        
        logger.info(f"\n=== BACKTEST COMPLETE ===")
        logger.info(f"Initial capital: {TOTAL_CAPITAL_START}")
        logger.info(f"Final capital: {final_capital:.2f}")
        logger.info(f"Gross P&L: {total_pnl:.2f}")
        logger.info(f"Total Charges: {total_charges:.2f}")
        logger.info(f"Net Profit: {net_profit:.2f}")
        logger.info(f"Number of trades: {len(results)}")
        
        # Print detailed final summary
        print("\n=== FINAL SUMMARY ===")
        print(f"Starting Capital: ₹{TOTAL_CAPITAL_START:,.2f}")
        print(f"Final Capital: ₹{final_capital:,.2f}")
        print(f"Gross P&L: ₹{total_pnl:,.2f}")
        print(f"Total Charges: ₹{total_charges:,.2f}")
        print(f"Net Profit: ₹{net_profit:,.2f}")

        print(f"\n=== TRADE STATISTICS ===")
        print(f"Total Trades: {len(results)}")
        print(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
        print(f"Average Holding Days: {avg_hold_days:.1f}")

        print("\n=== EXIT TYPE COUNTS ===")
        print(f"SL: {exit_types['SL']}")
        print(f"Force Exits (after {FORCE_EXIT_DAYS} days): {exit_types['FORCE_EXIT']}")
        print(f"Last Day Exits: {exit_types['TIME_STOP']}")
        print(f"T1: {exit_types['T1']}")
        print(f"T2: {exit_types['T2']}")
        print(f"T3: {exit_types['T3']}")
        print(f"SL After T1: {exit_types['T1_SL']}")
        print(f"SL After T2: {exit_types['T2_SL']}")
        
        # Save to CSV (without exit counts)
        df_res.to_csv(SWING_BACKTEST_CSV, index=False)
        
    else:
        logger.warning("No trades executed during backtest period")

if __name__ == '__main__':
    main()
