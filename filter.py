import pandas as pd
from datetime import datetime

# === Step 1: Load and sort data ===
df = pd.read_csv('swing_backtest_results.csv')

# Convert entry and exit dates to datetime using explicit format to avoid warnings
df['entry_date'] = pd.to_datetime(df['entry_date'], format="%Y-%m-%d")
df['exit_date'] = pd.to_datetime(df['exit_date'], format="%Y-%m-%d")

# Sort by entry date
df = df.sort_values(by='entry_date').reset_index(drop=True)

# === Step 2: Simulate Capital Allocation ===
total_capital = 200000  # Initial capital
available_capital = total_capital

# Store active trades (with exit date and capital used)
active_trades = []

# List to store selected trades
selected_trades = []

for i, row in df.iterrows():
    current_entry = row['entry_date']
    current_exit = row['exit_date']
    capital_needed = row['capital_used']

    # Step 1: Release capital from trades that have exited before or on current entry date
    active_trades = [t for t in active_trades if t['exit_date'] > current_entry]
    capital_in_use = sum(t['capital_used'] for t in active_trades)
    available_capital = total_capital - capital_in_use

    # Step 2: Check if we can take this trade
    if available_capital >= capital_needed:
        selected_trades.append(row)
        active_trades.append({'exit_date': current_exit, 'capital_used': capital_needed})
    # else: skip the trade

# === Step 3: Save the filtered trades ===
filtered_df = pd.DataFrame(selected_trades)
filtered_df.to_csv('filtered_trades.csv', index=False)

print("Filtered trades saved to 'filtered_trades.csv'")
