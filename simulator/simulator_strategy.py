import pandas as pd

# 加载数据
file_path = 'arima_prediction_comparison.csv'
data = pd.read_csv(file_path)

initial_capital = 10000  # 初始资金
capital = initial_capital
position = 0  # 当前持仓
buy_price = 0  # 最近一次买入价格

# 用于记录每次交易后的资本
capital_history = []

# 每50个数据点判断一次趋势，初始化为向上
trend = "upward"

for i in range(0, len(data), 50):
    sub_data = data.iloc[i:i + 50]

    trend_avg = sub_data['Forecast'].mean()

    # 判断趋势
    trend = "upward" if trend_avg > 0 else "downward"

    if i == 0 and trend == "upward":  # 初始且趋势为上升时，买入
        position = (capital * 0.5) / sub_data['avg_price'].iloc[0]
        position = round(position, 0)
        buy_price = sub_data['avg_price'].iloc[0]
        capital -= position * buy_price
    elif trend == "upward":  # 趋势为上升，考虑加仓
        last_forecast = data['Forecast'].iloc[i - 1] if i - 1 >= 0 else 0
        if sub_data['Forecast'].iloc[0] > last_forecast * 1.02:  # 预测值比上一次高2%
            additional_position = (capital * 0.1) / sub_data['avg_price'].iloc[0]
            position += additional_position
            capital -= additional_position * sub_data['avg_price'].iloc[0]

    # 检查是否需要卖出
    if position > 0:
        price_change = (sub_data['avg_price'].iloc[-1] - buy_price) / buy_price
        if trend == "downward" or price_change <= -0.035:  # 整体下跌或跌去3.5%
            capital += position * sub_data['avg_price'].iloc[-1]
            position = 0  # 全部卖出
            buy_price = 0

    capital_history.append(capital + position * sub_data['avg_price'].iloc[-1] if position > 0 else capital)

# 最终资金
final_capital = capital + position * data['avg_price'].iloc[-1] if position > 0 else capital
print(final_capital)
print(capital_history[-1])