import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error  ## 性能指标


#%%
# df = pd.read_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/result/sarimax_30s_prediction_comparison_2.csv')
# df = pd.read_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/result/sarimax_60s_prediction_comparison_2.csv')
df = pd.read_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/result/sarimax_300s_prediction_comparison_2.csv')

########################################################3
## 模型评价指标
'''
MSE, Mean Squared Error均方误差: 反映预测值与实际值之间差异的平均大小。MSE越小，表示模型预测越准确。
RMSE, Root Mean Squared Error均方根误差: 是MSE的平方根。它以与目标变量同样的单位衡量误差，通常用来比较不同模型的误差大小。
MAE, Mean Absolute Error平均绝对误差: 是预测值与实际观测值之间绝对差的平均值。与MSE相比，MAE对大的误差不那么敏感，提供了一个误差的直观量度。
MAPE, Mean Absolute Percentage Error平均绝对百分比误差: 表示预测误差作为实际观测值百分比的平均值。它可以提供误差的百分比表示，有助于了【解误差在总量中所占的比重】
'''
#%%
y_test = df['Actual']  ## 测试集中，l_t 特征的真实值
y_pred = df['Forecast']  ## 根据测试集，模型计算的l_t 特征的预测值
# 计算评价指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = 1 - mse / y_test.var()
print('r2: ', r2)
adjusted_r2 = 1 - (1 - r2) * (y_test.shape[0] - 1) / (y_test.shape[0] -1 -1)
print('adjusted_r2: ', adjusted_r2)
# 将指标存入字典
metrics = {
    'MSE': [mse],
    'RMSE': [rmse],
    'MAE': [mae],
    'MAPE': [mape],
    'r2': [r2],
    'adjusted_r2': [adjusted_r2]
}

# 创建数据框
metrics_df = pd.DataFrame(metrics)

print("metrics_df calculated！")
print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape}, r2: {r2}, adjusted_r2: {adjusted_r2}")