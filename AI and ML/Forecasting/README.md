This repository contains Python scripts for end-to-end sales data analysis, visualization, and time series modeling. The workflow demonstrates typical tasks required in retail or warehouse analytics using anonymized data and clear English comments.

Features
Loading and parsing raw sales data from CSV files
Data transformation and filtering for target warehouses
Time series analysis: extraction of dates, rolling means, holiday/weekend flags, etc.
Exploratory analytics and feature engineering

Visualization of key business dynamics (sales, stock, trends)

Correlation heatmaps and numeric statistics

Visualizations
Sales & stock over time: Line chart showing both sales and stock series
Rolling mean trend: Line chart for actual sales with rolling (e.g. 7-day) average overlay
Correlation heatmap: Pairwise correlations between numeric features

File Structure
main.py or analysis.py: Main data loading and analytics workflow (anonymized code)
sales_data.csv: Example anonymized CSV with warehouse sales data (format: Warehouse; date1; date2; ...)
(You may also include weather data, competitor files, etc. as CSV/XLSX)

Usage
Install required Python libraries:

bash
pip install pandas numpy matplotlib seaborn
Prepare your input data:
Place your cleaned and anonymized sales CSV in the working directory. The script expects the first column to be 'Warehouse' and the rest as date-value pairs.

Run the script:

bash
python main.py
View the generated plots:

Time series sales and stock visualization

7-day rolling mean trend

Correlation heatmap

Customization Tips
Update the selected_warehouses list in the script to filter analysis for different warehouse codes.

Adjust rolling window size as needed for trend smoothing.

Extend the script for forecasting (e.g., with SARIMAX, Random Forest, or XGBoost) as per your modeling requirements.

Notes
All scripts and data are fully anonymized: there are no sensitive business names or real warehouse identifiers.

Comments throughout the code are provided in English for better clarity and sharing.

Works with any similar warehouse/date/sales-stock formatted data.

