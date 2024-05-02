# import boto3
# import pandas as pd
# from io import BytesIO
# from io import StringIO  ## 为了后面存文件
#
#
# s3_client = boto3.client('s3')
#
# bucket_name = 'lob-data-processed'
# object_key = 'lob-data-processed/total_lob_30.csv'
#
# obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
# df = pd.read_csv(BytesIO(obj['Body'].read()))


#%%
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import dump
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from pmdarima import auto_arima
# import datetime

#%%
# df_1 = pd.read_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOBdata_process_weight/avg/total_lob_30.csv')
df = pd.read_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/whole_dataset_processed/total_lob_1s_k10_0.1.csv')
df = df.dropna()
# 解析日期
df['date'] = pd.to_datetime(df['date'])
# 将时间窗口转换为timedelta（时间窗口以秒为单位），并设置每天的起始时间为8:00
df['datetime'] = df['date'] + pd.to_timedelta('8 hours') + pd.to_timedelta(df['time_window'], unit='s')
# 设置新的日期时间为索引
df.set_index('datetime', inplace=True)


#%%
# df = df.iloc[:5000]  ## 限制数量
N = df.shape[0]
# 划分训练集和测试集
train = df.iloc[:int(N*0.8)]  # 训练集：除了最后N个观测点外的所有数据
test = df.iloc[int(N*0.8):N]  # 测试集：最后N个观测点
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
# print(stepwise_model.summary())
#%%
# 使用auto_arima寻找最佳SARIMAX模型参数
stepwise_model = auto_arima(train['l_t'],
                            exogenous=train[[ 'max_bid','min_ask','avg_price',
                                              'bid_level_diff', 'ask_level_diff',
                                              'bid_cumulative_depth', 'ask_cumulative_depth']],
                            start_p=1, start_q=1,
                            max_p=3, max_q=3, m=12,
                            start_P=0, seasonal=True,
                            d=1, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

# 打印选定模型的AIC
print(stepwise_model.aic())
print(stepwise_model.summary())
#%%
###########################################
# ADF 测试(ADF(Augmented Dickey-Fuller) 强迪基-福勒检验)——时间序列的平稳性
# def ADFtest(timeseries):
#     # 执行Augmented Dickey-Fuller测试
#     print('Results of Augmented Dickey-Fuller Test:')
#     dftest = adfuller(timeseries, autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
#     for key, value in dftest[4].items():
#         dfoutput['Critical Value (%s)' % key] = value
#     print(dfoutput)
#
# adf_result = ADFtest(train['l_t'])
#######################################
#%%
model = SARIMAX(train['l_t'],
                exog=train[[ 'max_bid','min_ask','avg_price',
                             # 'avg_price_change',
                             'bid_level_diff',
                             'ask_level_diff', 'bid_cumulative_depth', 'ask_cumulative_depth']],
                order=(1, 0, 2),
                seasonal_order=(2, 1, 0, 5))  # s需要根据数据的季节性周期进行调整,1天=86400秒
results = model.fit()

# 进行预测，注意在做出预测时也需要提供相应时期的外生变量
preds = results.forecast(steps=test.shape[0], exog=test[['max_bid','min_ask','avg_price',
                                                         # 'avg_price_change',
                                                         'bid_level_diff',
                                                         'ask_level_diff', 'bid_cumulative_depth',
                                                         'ask_cumulative_depth']])
preds.index = test.index

#%%
## save the model 存模型-- joblib ## from joblib import dump
# 指定路径并保存模型
model_path = 'E:\\Bristol_tb2\\mini_projectB\\mini_projectB_sample_0129_2024\\Problem B data\\JPMorgan_Set01\\whole_dataset_processed\\model\\SARIMAX_1s_k10.joblib'  ## 注意路径中的 双反斜杠
dump(results, model_path)

#### model.save('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/whole_dataset_processed/model/SARIMAX_1s_k10.h5')

#%%
preds_series = pd.Series(preds, index=test.index)
comparison_df = pd.DataFrame({'Actual': test['l_t'], 'Forecast': preds_series})
n = 500

#%%
plot_x = range(1, len(test)+1)
plt.figure(figsize=(10, 6))
plt.plot(plot_x[:n], test['l_t'][:n], label='Actual', color='red')
plt.plot(plot_x[:n], preds[:n], label='Forecast', color='blue')
plt.axhline(y=0, color='gray', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Avg Price change')
plt.title('Avg Price change Forecast vs Actual')
plt.legend()
plt.savefig('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/whole_dataset_processed/model/fc_(SARIMAX)_1s_k10.png')
plt.show() ## 看不了

##############################################
# # 将图像保存到BytesIO对象而不是文件
# img_data = BytesIO()
# plt.savefig(img_data, format='png', bbox_inches='tight')
# img_data.seek(0)  # 移动到流的开始位置
#
# # 指定图像文件名
# object_key = 'lob-data-processed/forecast_vs_actual(SARIMAX).png'
#
# # 上传图像数据到S3
# s3_client.upload_fileobj(img_data, bucket_name, object_key)
#
# print(f"Image uploaded to S3: s3://{bucket_name}/{object_key}")

########################################################
# 创建一个新的DataFrame来比较实际值和预测值

#%%
# 输出为csv文件，便于后续分析
comparison_df['avg_price'] = test['avg_price']
comparison_df.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/whole_dataset_processed/model/SARIMAX_1s_k10.csv', index=True)

####################################
# # 指定上传的文件名
# object_key = 'path/to/your/arima_prediction_comparison.csv'
#
# # 把DataFrame保存到一个字符串缓冲区中
# csv_buffer = StringIO()
# comparison_df.to_csv(csv_buffer, index=True)
#
# # 上传到S3
# s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=csv_buffer.getvalue())
#
# print(f"CSV uploaded to S3: s3://{bucket_name}/{object_key}")

########################################################3
'''
MSE, Mean Squared Error均方误差: 反映预测值与实际值之间差异的平均大小。MSE越小，表示模型预测越准确。
RMSE, Root Mean Squared Error均方根误差: 是MSE的平方根。它以与目标变量同样的单位衡量误差，通常用来比较不同模型的误差大小。
MAE, Mean Absolute Error平均绝对误差: 是预测值与实际观测值之间绝对差的平均值。与MSE相比，MAE对大的误差不那么敏感，提供了一个误差的直观量度。
MAPE, Mean Absolute Percentage Error平均绝对百分比误差: 表示预测误差作为实际观测值百分比的平均值。它可以提供误差的百分比表示，有助于了【解误差在总量中所占的比重】
'''
#%%
# 模型评价指标
y_test = comparison_df['Actual']  ## 测试集中，l_t 特征的真实值
y_pred = comparison_df['Forecast']  ## 根据测试集，模型计算的l_t 特征的预测值

# 计算评价指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# 将指标存入字典
metrics = {
    'MSE': [mse],
    'RMSE': [rmse],
    'MAE': [mae],
    'MAPE': [mape]
}

# 创建数据框
metrics_df = pd.DataFrame(metrics)

print("metrics_df calculated！")
# print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")
# print(f"AIC: {results.aic}, BIC: {results.bic}")   ##
metrics_df.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/whole_dataset_processed/model/SARIMAX_evaluation_1s_k10.csv', index=True)

# 保存到AWS的S3桶中
# ## 指定上传的文件名
# object_key = 'whole-dataset/cleaned-dataset/time-window-1s/LOB/SARIMAX_evaluation_1s_k10.csv'
# # 把DataFrame保存到一个字符串缓冲区中
# csv_buffer = StringIO()
# metrics_df.to_csv(csv_buffer, index=True)
# # 上传到S3
# s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=csv_buffer.getvalue())
# print(f"SARIMAX_evaluation uploaded to S3: s3://{bucket_name}/{object_key}")