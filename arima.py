import pandas as pd
from matplotlib import pyplot as plt


from statsmodels.tsa.statespace.sarimax import SARIMAX

#%%
df = pd.read_csv('process_data/UoB_Set01_2025-01-02LOBs.csv')
df = df.dropna()
start_date = pd.to_datetime('2025-01-02 08:00:00')
df['actual_datetime'] = start_date + pd.to_timedelta(df['time_window'], unit='s')

df.set_index('actual_datetime', inplace=True)

#%%
# 使用auto_arima寻找最佳SARIMAX模型参数
# stepwise_model = auto_arima(df['market_price'], exogenous=df[['time_window', 'bid_level_diff', 'ask_level_diff', 'bid_cumulative_depth', 'ask_cumulative_depth']],
#                             start_p=1, start_q=1,
#                             max_p=3, max_q=3, m=12,
#                             start_P=0, seasonal=True,
#                             d=None, D=1, trace=True,
#                             error_action='ignore',
#                             suppress_warnings=True,
#                             stepwise=True)
#
# # 打印选定模型的AIC
# #Best model:  ARIMA(1,0,2)(2,1,0)[12]
# print(stepwise_model.aic())

#%%
df = df.iloc[:5000]
N = df.shape[0]

# 划分训练集和测试集
train = df.iloc[:int(N*0.8)]  # 训练集：除了最后N个观测点外的所有数据
test = df.iloc[int(N*0.8):N]  # 测试集：最后N个观测点

model = SARIMAX(train['l_t'],
                exog=train[[ 'max_bid','min_ask','avg_price','avg_price_change','bid_level_diff', 'ask_level_diff', 'bid_cumulative_depth', 'ask_cumulative_depth']],
                order=(1, 0, 2),
                seasonal_order=(2, 1, 0, 5))  # s需要根据您数据的季节性周期进行调整,1天=86400秒
results = model.fit()

# 进行预测，注意在做出预测时也需要提供相应时期的外生变量
preds = results.forecast(steps=test.shape[0], exog=test[['max_bid','min_ask','avg_price','avg_price_change','bid_level_diff', 'ask_level_diff', 'bid_cumulative_depth', 'ask_cumulative_depth']])

#%%
preds_series = pd.Series(preds, index=test.index)
comparison_df = pd.DataFrame({'Actual': test['l_t'], 'Forecast': preds_series})
n = 100
plt.figure(figsize=(10, 6))
plt.plot(test.index[:n], test['l_t'][:n], label='Actual', color='red')
plt.plot(test.index[:n], preds[:n], label='Forecast', color='blue')
plt.axhline(y=0, color='gray', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Avg Price change')
plt.title('Avg Price change Forecast vs Actual')
plt.legend()
plt.savefig('forecast_vs_actual(SARIMAX).png')
plt.show()



# 创建一个新的DataFrame来比较实际值和预测值

