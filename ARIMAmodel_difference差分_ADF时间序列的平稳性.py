# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 创建一个函数来 检查数据的平稳性(ADF测试) & 输出 ADF测试的结果
def test_stationarity(timeseries):
    # 执行Augmented Dickey-Fuller测试
    print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    # 使用adfuller函数从statsmodels库执行Augmented Dickey-Fuller（ADF）测试。这个函数检查timeseries的平稳性。
    # autolag='AIC'指示adfuller函数自动选择滞后长度，以便AIC（赤池信息准则）最小化，这有助于提高检验的准确性。
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    # 使用dfouput存储ADF测试的前四个结果：检验统计量、P值、使用的滞后数和用于检验的观察数。
    # index=['']给出对应结果的索引标签。
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    # 这个循环遍历ADF测试结果中的 临界值字典（它是dftest的第五个元素）。
    # 对于每个显著性水平（例如1%，5%，10%），它将临界值添加到dfoutput序列中，使这些值也有易于理解的索引标签。
    print(dfoutput)

# 生成不平稳的时间序列
np.random.seed(0)
n = 100
x = np.cumsum(np.random.randn(n))
# type(x)
# Out[9]: numpy.ndarray

# 把它转换成Pandas的DataFrame格式
## 一列 时间序列数据 数值型
df = pd.DataFrame(x, columns=['value'])
# type(df)
# Out[11]: pandas.core.frame.DataFrame

# 检查原始数据的平稳性
test_stationarity(df['value'])
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                 -1.132038
# p-value                         0.702128
# #Lags Used                      0.000000
# Number of Observations Used    99.000000
# Critical Value (1%)            -3.498198
# Critical Value (5%)            -2.891208
# Critical Value (10%)           -2.582596
# dtype: float64
## ADF测试(时间序列数据的平稳性 测试)：
## 检验统计量 大于 临界值，而且 p值 >0.5 —— 数据可能是非平稳的

# 进行一阶差分
df['first_difference'] = df['value'] - df['value'].shift(1)

# 检查一阶差分后的数据的平稳性
test_stationarity(df['first_difference'].dropna())
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                -9.158402e+00
# p-value                        2.572287e-15   ## 表示2.572287乘以10的-15次方，即0.000000000000002572287。
# #Lags Used                     0.000000e+00
# Number of Observations Used    9.800000e+01
# Critical Value (1%)           -3.498910e+00
# Critical Value (5%)           -2.891516e+00
# Critical Value (10%)          -2.582760e+00
# dtype: float64
## 检验统计量 比 临界值更小，而且 p值远小于常用的显著性水平0.01,0.05,0.1 —— 可以认为有强有力的证据表明此时间序列数据是平稳的

# 进行二阶差分
df['second_difference'] = df['first_difference'] - df['first_difference'].shift(1)

# 检查二阶差分后的数据的平稳性
test_stationarity(df['second_difference'].dropna())
# Results of Augmented Dickey-Fuller Test:
# Test Statistic                 -5.459820
# p-value                         0.000003
# #Lags Used                     11.000000
# Number of Observations Used    86.000000
# Critical Value (1%)            -3.508783
# Critical Value (5%)            -2.895784
# Critical Value (10%)           -2.585038
# dtype: float64
## ## 检验统计量 比 临界值更小，而且 p值远小于常用的显著性水平0.01,0.05,0.1 —— 可以认为有强有力的证据表明此时间序列数据是平稳的

# 对于 差分的阶数order 的选择  how to choose the order of difference
# 对比分析：
# 效果：两种差分都成功使时间序列达到了平稳性，但一阶差分的检验统计量比二阶差分的更小，这意味着一阶差分的时间序列平稳性可能更为显著。
# 数据保留：一阶差分比二阶差分保留了更多的原始数据信息（二阶差分使部分数据信息丢失更多）。
# 选择原则：根据差分次数最小的原则，如果时间序列在进行一次差分后已经是平稳的，则没有必要再进行二次差分。
# 综上所述，考虑到一阶差分后的时间序列就已经显示出了平稳性，并且为了避免过度差分可能引入的问题，一阶差分是更好的选择。这样既能保证数据的平稳性，又能尽可能地保留时间序列的原始信息和结构。

# 可视化原始数据和差分后的数据
plt.figure(figsize=(12, 6))
plt.plot(df['value'], label='Original')
plt.plot(df['first_difference'], label='1st Order Difference')
plt.plot(df['second_difference'], label='2nd Order Difference')
plt.legend(loc='best')
plt.title('Original and Differenced Time Series')
plt.show()

