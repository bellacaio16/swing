import pandas as pd
import os

# === Setup paths ===
RESULT_CSV = 'swing_backtest_results.csv'
SUMMARY_DIR = 'summary'
os.makedirs(SUMMARY_DIR, exist_ok=True)

SUMMARY_DAILY = os.path.join(SUMMARY_DIR, 'summary_detailed.csv')
META_SUMMARY = os.path.join(SUMMARY_DIR, 'meta_summary.csv')

def main():
    df = pd.read_csv(RESULT_CSV, parse_dates=['entry_date', 'exit_date'])

    # === META STATS ===
    total_trades = len(df)
    total_pnl = df['pnl_total'].sum()

    win_trades = df[df['pnl_total'] > 0]
    loss_trades = df[df['pnl_total'] < 0]
    win_rate = len(win_trades) / total_trades * 100 if total_trades else 0
    avg_holding = df['holding_days'].mean()

    # Capital in use per day (active trades on each day)
    capital_by_day = []
    for date in pd.date_range(df['entry_date'].min(), df['exit_date'].max()):
        active = df[(df['entry_date'] <= date) & (df['exit_date'] >= date)]
        total = active['capital_used'].sum()
        capital_by_day.append((date, total))

    capital_daily_df = pd.DataFrame(capital_by_day, columns=['date', 'capital_in_use']).set_index('date')
    max_capital_in_use = capital_daily_df['capital_in_use'].max()

    # ROI on max capital in use
    realistic_roi = (total_pnl / max_capital_in_use) * 100 if max_capital_in_use else 0

    # === DAILY STATS BY EXIT DATE ===
    daily = df.groupby('exit_date').agg({
        'symbol': 'count',
        'pnl_total': 'sum',
        'capital_used': 'sum'
    }).rename(columns={
        'symbol': 'num_trades',
        'pnl_total': 'daily_pnl',
        'capital_used': 'daily_capital_used'
    })

    daily['cumulative_pnl'] = daily['daily_pnl'].cumsum()
    daily['peak'] = daily['cumulative_pnl'].cummax()
    daily['drawdown'] = daily['peak'] - daily['cumulative_pnl']
    daily['drawdown_pct'] = (daily['drawdown'] / daily['peak'].replace(0, 1)) * 100
    daily['roi_pct'] = (daily['daily_pnl'] / daily['daily_capital_used'].replace(0, 1)) * 100

    max_drawdown = daily['drawdown'].max()
    max_drawdown_pct = daily['drawdown_pct'].max()

    # === MERGE WITH TRADE DATA ===
    daily_reset = daily.reset_index()
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    merged = pd.merge(df, daily_reset, on='exit_date', how='left')
    merged.to_csv(SUMMARY_DAILY, index=False)

    # === META SUMMARY ===
    meta = {
        'Total Trades': total_trades,
        'Winning Trades': len(win_trades),
        'Losing Trades': len(loss_trades),
        'Breakeven/Zero Trades': total_trades - len(win_trades) - len(loss_trades),
        'Total PnL': round(total_pnl, 2),
        'Max Capital Used (Any Day)': round(max_capital_in_use, 2),
        'ROI (%) on Max Capital': round(realistic_roi, 2),
        'Win Rate (%)': round(win_rate, 2),
        'Avg Holding Days': round(avg_holding, 2),
        'Max Drawdown': round(max_drawdown, 2),
        'Max Drawdown (%)': round(max_drawdown_pct, 2)
    }

    pd.DataFrame([meta]).to_csv(META_SUMMARY, index=False)

    print(f"✅ Summary written to: {SUMMARY_DAILY}")
    print(f"📊 Meta summary written to: {META_SUMMARY}")

    
    # Load your CSV file
    df = pd.read_csv("swing_backtest_results.csv", parse_dates=["entry_date"])
    
    # Clean and normalize result column (optional)
    df['result'] = df['result'].str.strip()
    
    # Extract year-month for grouping
    df['month'] = df['entry_date'].dt.to_period('M')
    
    # Group by month
    summary = df.groupby('month').apply(lambda x: pd.Series({
        'Total_Trades': len(x),
        'SL_Hits': (x['result'] == 'SL').sum(),
        'Target_Hit1_or_2': ((x['hit1'] == True) | (x['hit2'] == True)).sum(),
        'QUALITY_EXIT': (x['result'] == 'QUALITY_EXIT').sum(),
        'REPLACE': (x['result'] == 'REPLACE').sum(),
        'FORCE_EXIT': (x['result'] == 'FORCE_EXIT').sum(),
        'Monthly_PnL': x['pnl_total'].sum(),
        'Capital_Used': x['capital'].sum(),
        'ROI_%': (x['pnl_total'].sum() / x['capital'].sum()) * 100 if x['capital'].sum() != 0 else 0
    })).reset_index()
    
    # Optional: Round values
    summary['Monthly_PnL'] = summary['Monthly_PnL'].round(2)
    summary['ROI_%'] = summary['ROI_%'].round(2)
    
    # Print summary
    print(summary)
    
    # Save to CSV if needed
    summary.to_csv("monthly_backtest_summary.csv", index=False)


if __name__ == '__main__':
    main()
