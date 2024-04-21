import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

stock_file = '/Users/fanxinwei/Desktop/code/git_repo/dsmp-2024-group20/input_data/total_lob_30.csv'
df = pd.read_csv(stock_file)
# 解析日期
df['date'] = pd.to_datetime(df['date'])
# 将时间窗口转换为timedelta（时间窗口以秒为单位），并设置每天的起始时间为8:00
df['datetime'] = df['date'] + pd.to_timedelta('8 hours') + pd.to_timedelta(df['time_window'], unit='s')
# 设置新的日期时间为索引
df.set_index('datetime', inplace=True)

price = df['avg_price']

train_data = price[:-100]
test_data = price[-100:]
train_data_diff = train_data.diff(1)
train_data_diff = train_data_diff.dropna()

model = ARIMA(train_data, order=(2, 1, 2))
results = model.fit()
forecast = results.forecast(steps=100)
forecast = results.forecast(steps=100)
forecast_df = pd.DataFrame(forecast.values, index=test_data.index)
forecast_df.to_csv('forecast_arima.csv')
