# 预加载模块
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

##########################
#%%
def read_csvfile_all3(directory):
    dataframes = []  # 用于存储每个 CSV 文件的 DataFrame
    file_names = os.listdir(directory)  # 获取目录中的所有文件名
    column_names = ['timestamp', 'price', 'quantity']
    for file_name in file_names:
        if file_name.endswith('.csv'):  # 先检查是否是 CSV 文件
            file_path = os.path.join(directory, file_name)  # 正确地构建文件路径
            data = pd.read_csv(file_path, header=None, names=column_names, usecols=[0, 1, 2])
            dataframes.append(data)  # 将 DataFrame 添加到列表中

    # 在所有 CSV 文件处理完成后，合并所有非空 DataFrame
    if dataframes:  # 确保列表不为空
        dataframe = pd.concat(dataframes, ignore_index=True)
        print('The type of data is', type(dataframe))
        return dataframe
    else:
        print('No CSV files found or DataFrame is empty.')
        return pd.DataFrame()  # 如果没有找到 CSV 文件或列表为空，则返回空的 DataFrame
#%%
##########################
#%%
# 测试
directory_trytry = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry/'
data_csv = read_csvfile_all3(directory_trytry)

# data_csv.info
## timestamp  price  quantity

#%%
# ## 直接读取现有的数据
# data_ask = pd.read_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed/LOB_ask/2025-01-02.csv')
# data_bid = pd.read_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed/LOB_bid/2025-01-02.csv')


#%%
##########################
# 计算加权平均价格——按照数量加权


## 计算加权平均价格——按照数量加权
def wavg(data_csv):
    data_csv['wavg_tapesprice'] = data_csv['price'] * data_csv['quantity']
    wavg_tapesprice = data_csv.groupby(['timestamp']).agg({'wavg_tapesprice': 'sum', 'quantity': 'sum'}).reset_index()
    wavg_tapesprice['weighted_avg_price'] = wavg_tapesprice['wavg_tapesprice'] / wavg_tapesprice['quantity']
    return wavg_tapesprice

wavg_tapesprice = wavg(data_csv)

#%%
##########################
# ADF 测试(ADF(Augmented Dickey-Fuller) 强迪基-福勒检验)——时间序列的平稳性
def ADFtest(timeseries):
    # 执行Augmented Dickey-Fuller测试
    print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

# 测试 test
adf_result_tapes= ADFtest(wavg_tapesprice['weighted_avg_price'])

## 结果 result of ADF
## 仅第一个文件
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                    -5.481303
# p-value                            0.000002
# #Lags Used                        44.000000
# Number of Observations Used    19712.000000
# Critical Value (1%)               -3.430682
# Critical Value (5%)               -2.861687
# Critical Value (10%)              -2.566848
# dtype: float64

## 前6个文件
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                -1.776423e+01
# p-value                        3.328447e-30
# #Lags Used                     6.900000e+01
# Number of Observations Used    1.046780e+05
# Critical Value (1%)           -3.430412e+00
# Critical Value (5%)           -2.861568e+00
# Critical Value (10%)          -2.566785e+00

#%%
##########################
# 加权平均价格——画图
# Tapes:  wavg_price diagram
plt.figure(figsize=(12, 6))
plt.plot(wavg_tapesprice['weighted_avg_price'], label='Tapes')
plt.legend(loc='best')
plt.ylabel('weighted_avg_price')
plt.title('Time Series')
plt.show()

#%%
##########################
# 画出自相关ACF和偏自相关PACF图像
y = wavg_tapesprice['weighted_avg_price']
plt.figure(figsize=(12, 6))
plot_acf(y, lags=5000)  # ACF图
plt.title('ACF of AR Time Series')
plt.show()
plot_pacf(y, lags=5000)  # PACF图
plt.title('PACF of AR Time Series')
plt.show()

#%%
##############################################################################
##############################################################################
# 使用 auto_arima 自动寻找最佳 ARIMA 模型
model_1 = pm.auto_arima(wavg_tapesprice['weighted_avg_price'], start_p=0, start_q=0,
                      test='adf',  # 使用ADF测试确定'd'
                      max_p=3, max_q=3,  # 设置p和q的最大值
                      m=1,  # 数据的季节性周期
                      d=None,  # 让模型自动确定最优的d
                      seasonal=False,  # 数据不包含季节性成分
                      stepwise=True,  # 使用逐步算法
                      suppress_warnings=True,  # 抑制警告信息
                      information_criterion='aic',  # 使用AIC选择最佳模型
                      trace=True)  # 打印搜索过程
model_1.fit
# 输出模型摘要
print(model_1.summary())

# ### about the warning :
# Warnings:
# [1] Covariance matrix calculated using the outer product of gradients (complex-step).
# 在自动ARIMA模型拟合过程中，协方差矩阵是通过梯度的外积（复步法）
# 意味着模型的参数估计是在较高的数值准确度下完成的
##协方差矩阵----衡量变量之间线性相关性。
## 在ARIMA模型中，协方差矩阵用于评估模型参数的不确定性和可信度。
## 外积梯度（OPG）----一种计算协方差矩阵的方法,适用于非线性模型的参数估计。
## 复步法--数值微分方法，用于提高参数估计的精度，可以减少计算中的数值错误。

# #————————————————————————————————————————————————————————
# model_1 = pm.auto_arima(wavg_tapesprice['weighted_avg_price'].iloc[:10000], start_p=0, start_q=0,
#                       test='adf',  # 使用ADF测试确定'd'
#                       max_p=3, max_q=3,  # 设置p和q的最大值
#                       m=1,  # 数据的季节性周期
#                       d=None,  # 让模型自动确定最优的d
#                       seasonal=False,  # 数据不包含季节性成分
#                       stepwise=True,  # 使用逐步算法
#                       suppress_warnings=True,  # 抑制警告信息
#                       information_criterion='aic',  # 使用AIC选择最佳模型
#                       trace=True)  # 打印搜索过程
# # 输出模型摘要
# print(model_1.summary())




##############################################################################
#%%
## 预测 Forecast  n_periods个点
n_periods = 8000
fc, confint = model_1.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(wavg_tapesprice['weighted_avg_price']), len(wavg_tapesprice['weighted_avg_price'])+n_periods)

##############################################################################
#%%
# 绘图  plotting
# 数据  make series
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

#%%

# 读取文件
column_names = ['timestamp', 'price', 'quantity']
file_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/Tapes/UoB_Set01_2025-01-03tapes.csv'
tapesprice_new = pd.read_csv(file_path, header=None, names=column_names, usecols=[0, 1, 2])
wavg_tapesprice_new = wavg(tapesprice_new)

#%%
##
plt.plot(wavg_tapesprice_new['weighted_avg_price'].iloc[:18000])
plt.plot(fc_series, color='red', label='fc_Tapes')
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k', alpha=.15)

plt.title("Final Forecast of WWW Usage")
plt.show()