import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from statsmodels.tsa.statespace.sarimax import SARIMAX

def mark_label(df,k,thresholds):
    #0:stay,1:up，2:down
    # df['m_minus'] = df['avg_price'].rolling(window=k).mean()
    df['m_plus'] = df['avg_price'].shift(-1).rolling(window=k, min_periods=1).mean().shift(-k+1)
    df['l_t'] = (df['m_plus'] - df['avg_price']) / df['avg_price']
    df['label'] = 0
    df.loc[df['l_t'] > thresholds, 'label'] = 1
    df.loc[df['l_t'] < -thresholds, 'label'] = 2

    return df

#%%
df = pd.read_csv('process_data/total_lob_30s.csv')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()
df = mark_label(df, 10, 0.1)



df.set_index('time', inplace=True)

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
df = df.iloc[:100000]
N = df.shape[0]

# 划分训练集和测试集
train = df.iloc[:int(N*0.8)]  # 训练集：除了最后N个观测点外的所有数据
test = df.iloc[int(N*0.8):N]  # 测试集：最后N个观测点

model = SARIMAX(train['l_t'],
                exog=train[[ 'avg_price','avg_price_change', 'bid_level_diff', 'ask_level_diff',
             'bid_ask_depth_diff']],
                order=(1, 0, 2),
                seasonal_order=(2, 1, 0, 2))  # s需要根据您数据的季节性周期进行调整,1天=86400秒
results = model.fit()

# 进行预测，注意在做出预测时也需要提供相应时期的外生变量
preds = results.forecast(steps=test.shape[0],
                         exog=test[['avg_price','avg_price_change',
                                    'bid_level_diff', 'ask_level_diff',
                                    'bid_ask_depth_diff']])

preds.index = test.index
#%%
preds_series = pd.Series(preds)
comparison_df = pd.DataFrame({
                              'Actual': test['l_t'],
                              'Forecast': preds_series,
                              'avg_price': test['avg_price']})
n = 100
time_start = 10
plt.figure(figsize=(10, 6))
plt.plot(test.index[time_start:n], test['l_t'][time_start:n], label='Actual', color='red')
plt.plot(test.index[time_start:n], preds[time_start:n], label='Forecast', color='blue')
# plt.axhline(y=0, color='gray', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Avg Price change')
plt.title('Avg Price change Forecast vs Actual(SARIMAX)')
plt.legend()
plt.savefig('forecast_vs_actual(SARIMAX).png')
plt.show()



