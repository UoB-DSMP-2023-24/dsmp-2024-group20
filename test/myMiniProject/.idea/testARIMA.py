'''''''''
model: ARIMA model (by ARIMA() and pm.auto_arima())
data : Tapes data with average price or weighted average price(using quantity)
       the best ARIMA model is ARIMA(1, 0, 0)

( include ADF test & ACF figure & PACF figure & forecast_original_data_figure )

question： Shall we try using log() to clean the data ? (After calculate the avg_price or wavg_price)
'''''''''
#%%
# import packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
import pmdarima as pm


#%%
## read the file
## 读csv文件
def read_csvfile_all3(directory):
    dataframes = []  # 用于存储每个 CSV 文件的DataFrame列表  List --save all the data from the same folder
    file_names = os.listdir(directory)  # 获取目录中的所有文件名  got every files' name
    column_names = ['timestamp', 'price', 'quantity']
    for file_name in file_names:
        if file_name.endswith('.csv'):  # 先检查是否是 CSV 文件  check whether the type of this file is ".csv"
            file_path = os.path.join(directory, file_name)  # 正确地构建文件路径  build file paths
            data = pd.read_csv(file_path, header=None, names=column_names, usecols=[0, 1, 2])
            dataframes.append(data)  # 将 DataFrame 添加到列表中  add it

    # 在所有 CSV 文件处理完成后，合并所有非空 DataFrame
    # After all CSV files have been processed, merge all non-empty DataFrames
    if dataframes:  # 确保列表不为空  if the List is not empty
        dataframe = pd.concat(dataframes, ignore_index=True)
        print('The type of data is', type(dataframe))
        return dataframe
    else:
        print('No CSV files found or DataFrame is empty.')
        return pd.DataFrame()
        # 如果没有找到 CSV 文件或列表为空，则返回空的 DataFrame
        # If no CSV file is found or the list is empty, return an empty DataFrame

#%%
# file_path and read it
## actually， data_csv is the "UoB_Set01_2025-01-02tapes.csv"
directory_trytry = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry/'
data_csv = read_csvfile_all3(directory_trytry)


# #%%
# # split the original data
# train = data_csv.iloc[:200]
# test = data_csv.iloc[201:220]

#%%
## 计算加权平均价格——按照数量加权
## Calculate weighted average price - weighted by quantity
def wavg(data_csv):
    data_csv['wavg_tapesprice'] = data_csv['price'] * data_csv['quantity']
    wavg_tapesprice = data_csv.groupby(['timestamp']).agg({'wavg_tapesprice': 'sum', 'quantity': 'sum'}).reset_index()
    wavg_tapesprice['weighted_avg_price'] = wavg_tapesprice['wavg_tapesprice'] / wavg_tapesprice['quantity']
    return wavg_tapesprice

wavg_tapesprice = wavg(data_csv)

#%%
### 选点数据 & 划分数据集
### Select some data from the dataset to fit and test model
# split the weight-average-price data from wavg_tapesprice
train_wavg = wavg_tapesprice[:200]
test_wavg = wavg_tapesprice[200:220]

#%%
## 计算平均价格
## Calculate average price - group by timestamp
def avgprice(data_dataframe):
    avg_price = data_dataframe.groupby(['timestamp']).agg({'price': 'mean', 'quantity': 'mean'}).reset_index()
    return avg_price

avg_tapesprice = avgprice(data_csv)

# split the data
train_avg = avg_tapesprice[:200]
test_avg = avg_tapesprice[200:220]

#%%
# ADF test (ADF(Augmented Dickey-Fuller) 强迪基-福勒检验)
def ADFtest(timeseries):
    # 执行Augmented Dickey-Fuller测试
    print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

# 测试 test
adf_result_tapes= ADFtest(wavg_tapesprice['weighted_avg_price'].iloc[:500])
# # restult
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                -7.820398e+00
# p-value                        6.694158e-12
# #Lags Used                     1.000000e+00
# Number of Observations Used    4.980000e+02
# Critical Value (1%)           -3.443549e+00
# Critical Value (5%)           -2.867361e+00
# Critical Value (10%)          -2.569870e+00
# dtype: float64

#%%
adf_result_train_wavg= ADFtest(train_wavg['weighted_avg_price'])
#%%
adf_result_train_avg= ADFtest(train_avg['price'])

## result of train_wavg data
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                -7.820398e+00
# p-value                        6.694158e-12
# #Lags Used                     1.000000e+00
# Number of Observations Used    4.980000e+02
# Critical Value (1%)           -3.443549e+00
# Critical Value (5%)           -2.867361e+00
# Critical Value (10%)          -2.569870e+00
# dtype: float64

## result of train_avg data
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                -6.432011e+00
# p-value                        1.688296e-08
# #Lags Used                     0.000000e+00
# Number of Observations Used    1.990000e+02
# Critical Value (1%)           -3.463645e+00
# Critical Value (5%)           -2.876176e+00
# Critical Value (10%)          -2.574572e+00
# dtype: float64


#%%
##########################
# plot the figure
## weighted_avg_price of train_wavg data
# 加权平均价格——画图
plt.figure(figsize=(12, 6))
plt.plot(train_wavg['weighted_avg_price'], label='train_wavg_price')
plt.legend(loc='best')
plt.ylabel('weighted_avg_price of train_wavg data')
plt.title('train_wavg data')
plt.show()

#%%
## avg_price of train_avg data
plt.figure(figsize=(12, 6))
plt.plot(train_avg['price'], label='train_avg_price')
plt.legend(loc='best')
plt.ylabel('avg_price of train_avg data')
plt.title('train_avg data')
plt.show()

#%%
##########################
# 画出自相关ACF和偏自相关PACF图像
# Draw autocorrelation ACF and partial autocorrelation PACF images
## train_wavg data
y = train_wavg['weighted_avg_price']
plt.figure(figsize=(12, 6))
plot_acf(y, lags=40)  # ACF图
plt.title('ACF of train_wavg AR Time Series')
plt.show()
plot_pacf(y, lags=40)  # PACF图
plt.title('PACF of train_wavg AR Time Series')
plt.show()
## based on the figures -> AR(1) is suitable -> for ARIMA : p=1，d=0，q=0.

#%%
## train_avg data
y = train_avg['price']
plt.figure(figsize=(12, 6))
plot_acf(y, lags=40)  # ACF图
plt.title('ACF of train_avg AR Time Series')
plt.show()
plot_pacf(y, lags=40)  # PACF图
plt.title('PACF of train_avg AR Time Series')
plt.show()


## based on the figures -> AR(1) is suitable -> for ARIMA : p=1，d=0，q=0.

#%%
##############################################################################
##############################################################################
# implement the ARIMA model  ARIMA(1, 0, 0)

##############################################################################
# using train_avg data
y = train_avg['price']
y_test = test_avg['price']
## delete the below
# keys_traindata = list(y.keys)
# keys_testdata = list(y_test.keys)
##############################################################################
# # using train_wavg data
# y = train_wavg['wavg_tapesprice']
# y_test = test_wavg['weighted_avg_price']
## delete the below
# keys_traindata = list(y.keys)
# keys_testdata = list(y_test.keys)
##############################################################################
#%%
## fit
model = ARIMA(y, order=(1, 0, 0))
model_fit = model.fit()

#%%
# Check whether the residuals残差 are white noise (i.e. uncorrelated)
# 模型检测———— 是否有白噪声(自相关性)
residuals = model_fit.resid
plot_acf(residuals)
plt.title('Residuals of train_wavg AR')
plt.show()

#%%
## forecast-- 20 steps
## 预测未来20个时间点
forecast = model_fit.forecast(steps=20)
print(forecast)

fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, label = 'test data')
plt.plot(forecast, label = 'forecast')
plt.legend(loc='best')
plt.xlabel('timestamps')
plt.ylabel('price')
plt.title('ARIMA: forcast & original price')
plt.show()

# ### steps depends on the len(y), y is the training data
# ### 灵活步长， 比训练数据的长度多20
# plot_predict(model_fit, start=1, end=len(y) + 20)
# plt.show()

##############################################################################
## remember go back and train model ARIMA(1, 0, 0) by using train_wavg data
##############################################################################



##############################################################################
### remember to check y is base on with training data
##############################################################################
#%%
## using auto_arima to find the best ARIMA model
## import pmdarima as pm
## 自动寻找最佳ARIMA模型

### remember to check y is base on with training data: train_avg ? or train_wavg?
auto_model = pm.auto_arima( y, start_p=0, start_q=0,
                            test='adf',
                            max_p=5, max_q=5,
                            seasonal=False,   ## if seasonal=True, need to add "start_P = " and "D="
                            # although "False" is been choosen, must set the "m="( but "m=" does't work)
                            suppress_warnings= 'True',
                            information_criterion= 'aic', ## using aic to choose the best model
                            error_action='ignore',
                            # stepwise= True,    ## “True”(default) may cannot find the best way, but "False" will waste more time.
                            trace= True   ## print the detail of the searching process
                            )
auto_model.fit

print(auto_model.summary())

# the trace of the model --- train_avg data
# Performing stepwise search to minimize aic
# Best model:  ARIMA(1,0,0)(0,0,0)[0] intercept
# Total fit time: 1.095 seconds
# Out[68]: <bound method ARIMA.fit of ARIMA(order=(1, 0, 0), scoring_args={}, suppress_warnings='True')>

## the summary of the auto_model
# Performing stepwise search to minimize aic
#  ARIMA(0,0,0)(0,0,0)[0]             : AIC=2797.899, Time=0.00 sec
#  ARIMA(1,0,0)(0,0,0)[0]             : AIC=inf, Time=0.02 sec
#  ARIMA(0,0,1)(0,0,0)[0]             : AIC=inf, Time=0.02 sec
#  ARIMA(1,0,1)(0,0,0)[0]             : AIC=996.727, Time=0.03 sec
#  ARIMA(2,0,1)(0,0,0)[0]             : AIC=984.051, Time=0.09 sec
#  ARIMA(2,0,0)(0,0,0)[0]             : AIC=inf, Time=0.02 sec
#  ARIMA(3,0,1)(0,0,0)[0]             : AIC=984.418, Time=0.13 sec
#  ARIMA(2,0,2)(0,0,0)[0]             : AIC=999.647, Time=0.15 sec
#  ARIMA(1,0,2)(0,0,0)[0]             : AIC=996.562, Time=0.04 sec
#  ARIMA(3,0,0)(0,0,0)[0]             : AIC=inf, Time=0.03 sec
#  ARIMA(3,0,2)(0,0,0)[0]             : AIC=inf, Time=0.14 sec
#  ARIMA(2,0,1)(0,0,0)[0] intercept   : AIC=970.331, Time=0.11 sec
#  ARIMA(1,0,1)(0,0,0)[0] intercept   : AIC=968.325, Time=0.17 sec
#  ARIMA(0,0,1)(0,0,0)[0] intercept   : AIC=1008.148, Time=0.02 sec
#  ARIMA(1,0,0)(0,0,0)[0] intercept   : AIC=968.268, Time=0.03 sec
#  ARIMA(0,0,0)(0,0,0)[0] intercept   : AIC=1079.534, Time=0.00 sec
#  ARIMA(2,0,0)(0,0,0)[0] intercept   : AIC=968.350, Time=0.04 sec
# Best model:  ARIMA(1,0,0)(0,0,0)[0] intercept
# Total fit time: 1.065 seconds
#                                SARIMAX Results
# ==============================================================================
# Dep. Variable:                      y   No. Observations:                  200
# Model:               SARIMAX(1, 0, 0)   Log Likelihood                -481.134
# Date:                Mon, 19 Feb 2024   AIC                            968.268
# Time:                        20:36:31   BIC                            978.162
# Sample:                             0   HQIC                           972.272
#                                 - 200
# Covariance Type:                  opg
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# intercept     90.1915     15.676      5.753      0.000      59.466     120.917
# ar.L1          0.6566      0.060     10.989      0.000       0.540       0.774
# sigma2         7.1763      0.674     10.646      0.000       5.855       8.497
# ===================================================================================
# Ljung-Box (L1) (Q):                   0.78   Jarque-Bera (JB):                11.94
# Prob(Q):                              0.38   Prob(JB):                         0.00
# Heteroskedasticity (H):               0.66   Skew:                             0.46
# Prob(H) (two-sided):                  0.10   Kurtosis:                         3.77
# ===================================================================================
# Warnings:
# [1] Covariance matrix calculated using the outer product of gradients (complex-step).

## Best model:  ARIMA(1,0,0)

#%%
## 预测  forecast dataponit-- number : n_periods
##
n_periods = 20
fc_auto = pd.DataFrame(auto_model.predict(n_periods=n_periods), columns=['price'])
# fc_auto['timestamp'] = data_csv['timestamp'][201:221].values  ## [.values] 在数据框中，新建一列》保证这一列的数据和原本其他列的数据所在行数一致
print(fc_auto)

#%%
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, label = 'test data')
plt.plot(fc_auto['price'], label = 'forecast')
plt.legend(loc='best')
plt.xlabel('timestamps')
plt.ylabel('price')
plt.title('auto_ARIMA : forcast & original price')
plt.show()

#%%
##############################################################################
##############################################################################
## other kinds way to clean data
### try to use log() function ????
### we can discuss it later