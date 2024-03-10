#%%
import pandas as pd
import numpy as np
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%%
path = 'E:/Bristol_tb2/mini_projectB/brunch_my/dsmp-2024-group20/2025-01-02_feature.csv'
df = pd.read_csv(path)
df.drop(columns='label', inplace=True)

print(df.head(5))
print(df.keys())
#%%
## 初步判断共线性
# 计算相关系数矩阵
# calculate the correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)

name = '2025-01-02_feature_cm.csv'
dire_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01'
output_path = os.path.join(dire_path, name)  ## import os # 完整的输出路径
correlation_matrix.to_csv(output_path, index=False)

# 通常采用的基准来判断相关性的强度：
# 弱相关性：如果相关系数的绝对值在0到0.3（或-0.3）之间。
# 中等相关性：如果相关系数的绝对值在0.3到0.7（或-0.3到-0.7）之间。
# 强相关性：如果相关系数的绝对值在0.7到1.0（或-0.7到-1.0）之间。


#%%
## 共线性检测：方差膨胀因子（VIF） -- 更精确地判断多重共线性
## （Variance Inflation Factor, VIF）
# VIF值大于5（或有时用10作为阈值）表明高度的多重共线性，这意味着该变量可以由其他一个或多个变量预测

df_VIFtest = df.drop(columns=['time_window',
                              'avg_price_change', 'bid_ask_depth_diff', 'next_avg_price_change'], inplace=False)

# 计算VIF
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns

#%%
# 添加VIF分数
### 【报错--修改】:使用pandas的dropna()方法删除含有缺失值的行
### 使用numpy的isinf()函数检查无限值
### VIF阈值 5 (或者10)

vif_data["VIF"] = [variance_inflation_factor(df.dropna().values, i) for i in range(len(df.columns))]



print(vif_data)

#                   feature           VIF
# 0             time_window  4.679049e+00
# 1                 max_bid           inf
# 2                 min_ask           inf
# 3    bid_cumulative_depth  8.256610e+01
# 4    ask_cumulative_depth  9.359427e+01
# 5               avg_price           inf
# 6        avg_price_change  1.240531e+00
# 7          bid_level_diff           inf
# 8          ask_level_diff           inf
# 9      bid_ask_depth_diff  2.598293e+01
# 10  next_avg_price_change  1.335917e+00
# 11                 m_plus  2.217258e+04
# 12                    l_t  5.094513e+01


#                   feature           VIF
# 0             time_window  4.679049e+00
# 1                 max_bid           inf
# 2                 min_ask           inf
# 3    bid_cumulative_depth  8.256610e+01
# 4    ask_cumulative_depth  9.359427e+01
# 5               avg_price           inf
# 6        avg_price_change  1.240531e+00
# 7          bid_level_diff           inf
# 8          ask_level_diff           inf
# 9      bid_ask_depth_diff  2.598293e+01
# 10  next_avg_price_change  1.335917e+00
# 11                 m_plus  2.217258e+04
# 12                    l_t  5.094513e+01


#%%
# 条件指数（Condition Index）




#%%
## 显著性检验  -- 没有实操的价值


# import pandas as pd
# from scipy.stats import pearsonr
#
# # 假设df是一个pandas DataFrame，并且有两列数值数据，我们想要检查这两列之间的相关性
# # 例子中的列名分别为'column1'和'column2'
#
# # 计算相关系数和p值
# # ## ['time_window', 'max_bid', 'min_ask', 'bid_cumulative_depth',
# #        'ask_cumulative_depth', 'avg_price', 'avg_price_change',
# #        'bid_level_diff', 'ask_level_diff', 'bid_ask_depth_diff',
# #        'next_avg_price_change', 'm_plus', 'l_t']
# column1 = 'ask_level_diff'
# column2 = 'max_bid'
#
# df__ = df.dropna()
# corr_coefficient, p_value = pearsonr(df__[column1], df__[column2])
#
# print(f"Pearson Correlation Coefficient: {corr_coefficient}")
# print(f"P-value: {p_value}")
#
# # 判断显著性
# alpha = 0.05  # 常用的显著性水平为0.05
# if p_value < alpha:
#     print("两个变量之间存在统计学显著的相关性")
# else:
#     print("两个变量之间不存在统计学显著的相关性")

