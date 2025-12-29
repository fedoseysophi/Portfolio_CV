import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# === 1. Load and parse sales data ===
sales_file = "/content/sales_warehouse.csv"
with open(sales_file, 'r', encoding='utf-8') as file:
    sales_lines = file.readlines()

# Remove BOM and parse date row with format dd-mm
raw_dates = sales_lines[0].replace('\ufeff', '').strip().split(';')
dates = [d for d in raw_dates if '-' in d]

# Automatically determine the year
parsed_dates = []
current_year = 2024
last_month = 0

for date_str in dates:
    try:
        day, month = map(int, date_str.split('-'))
        if last_month == 12 and month == 1:
            current_year += 1
        dt = pd.to_datetime(f"{day:02d}-{month:02d}-{current_year}", format='%d-%m-%Y')
        parsed_dates.append(dt)
        last_month = month
    except:
        parsed_dates.append(None)

date_map = pd.Series(parsed_dates)

# Parse data rows
parsed_data = []
for line in sales_lines[2:]:
    parts = line.strip().split(';')
    warehouse = parts[0]
    values = parts[1:]
    for i in range(0, len(values) - 1, 2):
        idx = i // 2
        if idx < len(date_map):
            sales = values[i]
            stock = values[i + 1]
            parsed_data.append((warehouse, date_map[idx], sales, stock))

# === 2. Create DataFrame for selected warehouses ===
df_warehouse = pd.DataFrame(parsed_data, columns=['Warehouse', 'Date', 'Sales', 'Stock'])

# List of required warehouses
selected_warehouses = [
    'KHORUGVINO_RFC',
    'GRIVNO_RFC',
    'SOFYINO_RFC',
    'PUSHKINO_1_RFC',
    'ZHUKOVSKY_RFC',
    'PETROVSKOE_RFC',
    'NOGINSK_RFC',
    'TVER_RFC'
]

# Filter DataFrame by selected warehouses
df_warehouse = df_warehouse[df_warehouse['Warehouse'].isin(selected_warehouses)]

# Convert data types
df_warehouse['Sales'] = pd.to_numeric(df_warehouse['Sales'], errors='coerce')
df_warehouse['Stock'] = pd.to_numeric(df_warehouse['Stock'], errors='coerce')
df_warehouse['Date']  = pd.to_datetime(df_warehouse['Date'])

# === 3. Load Excel file with additional columns ===
excel_df = pd.read_excel("/content/OZON_sales.xlsx")
excel_df['Date'] = pd.to_datetime(excel_df['Date'], errors='coerce')
excel_df = excel_df[['Date', 'PriceWithOzonCard', 'Revenue', 'Comments', 'Rating']]

# === 4. Load weather data ===
weather_df = pd.read_csv("/content/moscow.csv")
weather_df['Date'] = pd.to_datetime(weather_df['date'], errors='coerce')
weather_df = weather_df[['Date', 'tavg', 'prcp']]

# === 5. Merge DataFrames ===
df_merged = df_warehouse.merge(excel_df, on='Date', how='left')
df_final = df_merged.merge(weather_df, on='Date', how='left')

# === 6. Save result to CSV ===
df_final.to_csv("final_dataset.csv", index=False, encoding='utf-8-sig')

# Preview first 100 rows of the final dataset
print(df_final.head(100))

# Check shape of the final DataFrame
print(df_final.shape)

# === Add other blocks similarly... ===

# Example: Feature engineering block for calendar, lag, rolling stats, etc.
df = df_final.copy()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# 1. Calendar features
df['day_of_week'] = df.index.dayofweek        # 0=Monday ... 6=Sunday
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['day'] = df.index.day
df['day_of_year'] = df.index.dayofyear

# 2. Lag features for sales
for lag in (1, 7, 14, 30):
    df[f'sales_lag_{lag}'] = df['Sales'].shift(lag)

# 3. Rolling statistics for sales
for window in (7, 14, 30):
    df[f'sales_roll_mean_{window}'] = df['Sales'].rolling(window).mean()
    df[f'sales_roll_std_{window}'] = df['Sales'].rolling(window).std()

# 4. Price features
df['price_lag_1'] = df['PriceWithOzonCard'].shift(1)
df['price_lag_7'] = df['PriceWithOzonCard'].shift(7)
df['price_pct_1d'] = df['PriceWithOzonCard'].pct_change(1)
df['price_pct_7d'] = df['PriceWithOzonCard'].pct_change(7)
df['price_ratio_1d'] = df['PriceWithOzonCard'].shift(1) / df['PriceWithOzonCard']

# 5. Weather features
df['prcp_sum_7d'] = df['prcp'].rolling(7).sum()
df['tavg_mean_7d'] = df['tavg'].rolling(7).mean()
df['tavg_anom_30d'] = df['tavg'] - df['tavg'].rolling(30).mean()

# 6. Trend and Fourier features for yearly seasonality
df['time_idx'] = np.arange(len(df))
df['sin_365'] = np.sin(2 * np.pi * df['time_idx'] / 365)
df['cos_365'] = np.cos(2 * np.pi * df['time_idx'] / 365)

# 7. Sales slope over the last 7 days
def slope(x):
    if np.isnan(x).any():
        return np.nan
    return np.polyfit(np.arange(len(x)), x, 1)[0]

df['sales_slope_7d'] = df['Sales'].rolling(7).apply(slope, raw=True)

# 8. Monthly seasonality features
df['month_avg_sales'] = df['Sales'].groupby(df.index.month).transform('mean')
month_dummies = pd.get_dummies(df['month'], prefix='month')
df = pd.concat([df, month_dummies], axis=1)

# 9. Remove rows with NaN from lag/rolling ops
df.fillna(0, inplace=True)

# Continue adapting all code blocks in this pattern

# === All comments are now in English and variable names are anonymized. ===
