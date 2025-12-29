import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load and parse sales data from CSV file ===
file_path = 'sales_data.csv'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Extract dates from the header row
raw_dates = lines[0].strip().split(';')
dates = [d for d in raw_dates if '-' in d]
parsed_dates = []
year = 2024
prev_month = 0

for d in dates:
    day, month = map(int, d.split('-'))
    if prev_month == 12 and month == 1:
        year += 1
    try:
        dt = pd.to_datetime(f"{day:02d}-{month:02d}-{year}", format='%d-%m-%Y')
        parsed_dates.append(dt)
        prev_month = month
    except:
        parsed_dates.append(None)

date_map = pd.Series(parsed_dates)

# Parse the remaining rows containing sales and stock data
parsed_data = []
for line in lines[2:]:
    parts = line.strip().split(';')
    warehouse = parts[0]
    values = parts[1:]
    for i in range(0, len(values) - 1, 2):
        idx = i // 2
        if idx < len(date_map):
            sales = values[i]
            stock = values[i + 1]
            parsed_data.append((warehouse, date_map[idx], sales, stock))

# Create main DataFrame
df = pd.DataFrame(parsed_data, columns=['Warehouse', 'Date', 'Sales', 'Stock'])

# Select target warehouses (example names)
selected_warehouses = [
    'WAREHOUSE_A', 'WAREHOUSE_B', 'WAREHOUSE_C'
]
df = df[df['Warehouse'].isin(selected_warehouses)]

# Convert columns to numeric types
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df['Stock'] = pd.to_numeric(df['Stock'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'])

# === 2. Basic statistics and time series plot ===
print(df.head())
print(df.describe())

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Sales'], label='Sales', marker='o')
plt.plot(df['Date'], df['Stock'], label='Stock', marker='s')
plt.title('Sales and Stock Dynamics')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 3. Rolling mean trend visualization ===
df['sales_roll7'] = df['Sales'].rolling(7, min_periods=1).mean()
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Sales'], label='Actual Sales', alpha=0.6)
plt.plot(df['Date'], df['sales_roll7'], label='7-day Rolling Mean', color='red')
plt.legend()
plt.title('Sales & 7-day Trend')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === 4. Correlation heatmap between numeric features ===
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# The script can be expanded further for forecasting, feature engineering, competitor analysis, etc.
