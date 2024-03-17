import pandas as pd

# 加载数据
file_path = 'simulator/arima_prediction_comparison.csv'
data = pd.read_csv(file_path)

initial_funds = 10000
funds = initial_funds
stocks = 0
buy_sell_points = []  # List to keep track of buy/sell actions

for index, row in data.iterrows():
    if row['Forecast'] > 0 and funds >= row['avg_price']:  # Buy condition
        funds -= row['avg_price']  # Deduct the price of one stock from funds
        stocks += 1  # Increase stock count
        buy_sell_points.append('Buy')
    elif row['Forecast'] < 0 and stocks > 0:  # Sell condition
        funds += row['avg_price']  # Add the price of one stock to funds
        stocks -= 1  # Decrease stock count
        buy_sell_points.append('Sell')
    else:
        buy_sell_points.append('Hold')

# Add the buy/sell points to the dataframe
data['Action'] = buy_sell_points

output_file_path = 'simulator/arima_prediction_with_trades.csv'
data.to_csv(output_file_path, index=False)

# Calculate the value of remaining stocks at the last available price
final_stock_value = stocks * data.iloc[-1]['avg_price']
total_value = funds + final_stock_value
profit = total_value - initial_funds

print(funds, stocks, final_stock_value, total_value, profit)


#%%
import matplotlib.pyplot as plt

# Load the updated CSV to ensure we're working with the correct data
data_with_trades = pd.read_csv('simulator/arima_prediction_with_trades.csv')

# Plotting with vertical lines for buy/sell points on the first 100 points
fig, ax = plt.subplots(figsize=(14, 8))

# Limit data to first 100 points for plotting
limited_data = data_with_trades.iloc[:100]

# Adjusting the plot to display fewer x-axis labels for clarity
fig, ax = plt.subplots(figsize=(14, 8))

# Plot Actual and Forecast values
ax.plot(limited_data['actual_datetime'], limited_data['Actual'], label='Actual', color='blue', alpha=0.6)
ax.plot(limited_data['actual_datetime'], limited_data['Forecast'], label='Forecast', color='green', alpha=0.6)

# Add vertical lines for buy/sell points within the first 100 points
buy_points_limited = limited_data[limited_data['Action'] == 'Buy']
sell_points_limited = limited_data[limited_data['Action'] == 'Sell']

# Add vertical lines for buy/sell points
for _, row in buy_points_limited.iterrows():
    ax.axvline(x=row['actual_datetime'], color='lime', label='Buy Point', linestyle='--')

for _, row in sell_points_limited.iterrows():
    ax.axvline(x=row['actual_datetime'], color='red', label='Sell Point', linestyle='--')

# Add a solid black line for points where value equals 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=2)

# Set x-axis labels to display fewer labels to avoid crowding
ticks_to_show = limited_data.index[::10]  # Show every 10th label
ax.set_xticks(ticks_to_show)
ax.set_xticklabels(limited_data['actual_datetime'].iloc[::10], rotation=45)

# Beautify the plot
plt.xlabel('Datetime')
plt.ylabel('Avg price change')
plt.title('Actual vs. Forecast with Trade Signals (First 100 Points)')
# Handle duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left')
plt.tight_layout()

plt.show()