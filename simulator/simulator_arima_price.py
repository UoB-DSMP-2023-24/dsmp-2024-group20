import pandas as pd

# 加载数据
file_path = '/Users/fanxinwei/Desktop/code/git_repo/dsmp-2024-group20/model/arima/forecast_arima.csv'
data = pd.read_csv(file_path)

initial_funds = 10000
funds = initial_funds
stocks = 0
buy_sell_points = []  # List to keep track of buy/sell actions

# We skip the first row initially because there is no previous forecast to compare with
for index in range(1, len(data)):
    current_forecast = data.loc[index, 'Forecast']
    previous_forecast = data.loc[index - 1, 'Forecast']
    avg_price = data.loc[index, 'tapes']

    if current_forecast > previous_forecast and funds >= avg_price:  # Buy condition
        funds -= avg_price  # Deduct the price of one stock from funds
        stocks += 1  # Increase stock count
        buy_sell_points.append('Buy')
    elif current_forecast < previous_forecast and stocks > 0:  # Sell condition
        funds += avg_price  # Add the price of one stock to funds
        stocks -= 1  # Decrease stock count
        buy_sell_points.append('Sell')
    else:
        buy_sell_points.append('Hold')

# Insert 'Hold' for the first row as there's no action to take
buy_sell_points.insert(0, 'Hold')

# Add the buy/sell points to the dataframe
data['Action'] = buy_sell_points

output_file_path = '/Users/fanxinwei/Desktop/code/git_repo/dsmp-2024-group20/model/arima/arima_trades.csv'
data.to_csv(output_file_path, index=False)

# Calculate the value of remaining stocks at the last available price
final_stock_value = stocks * data.iloc[-1]['tapes']
total_value = funds + final_stock_value
profit = total_value - initial_funds

print(funds, stocks, final_stock_value, total_value, profit)

