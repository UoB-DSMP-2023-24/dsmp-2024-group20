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

# 计算VIF
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns

#%%
# 添加VIF分数
### 【报错--修改】:使用pandas的dropna()方法删除含有缺失值的行
### 使用numpy的isinf()函数检查无限值

vif_data["VIF"] = [variance_inflation_factor(df.dropna().values, i) for i in range(len(df.columns))]



print(vif_data)



#%%
# 条件指数（Condition Index）

