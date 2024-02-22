#%%
import numpy as np
import pandas as pd


def aggregate_data(df, second_column, aggregation_rules,second):
    """
    聚合数据函数。

    参数:
    - df: pandas DataFrame，包含原始数据。
    - second_column: string，包含秒数的列名。
    - aggregation_rules: dict，定义要聚合的列及其聚合方法。
    - second: int，时间窗口大小,按照多少秒进行聚合。

    返回:
    - 聚合后的DataFrame。
    """
    # 计算时间窗口
    df['time_window'] = (df[second_column] // second) * second

    # 根据时间窗口聚合数据
    aggregated_df = df.groupby('time_window').agg(aggregation_rules).reset_index()
### 无法满足计算加权平均值的要求，对该函数进行优化，增加下面的代码：


    return aggregated_df

#%%
# #可以使用自定义方法
# def divide_by_2(series):
#     return np.sum(series)/ 2

## Calculate weighted average price - weighted by quantity
# 定义 wavg_aggrule 函数
def wavg_aggrule(group, price_col, quantity_col):
    """
    计算加权平均价格。

    参数:
    - group: DataFrame分组
    - price_col: 价格列名
    - quantity_col: 数量列名

    返回:
    - 加权平均价格的DataFrame
    """
    d = {}
    d['weighted_avg_price'] = (group[price_col] * group[quantity_col]).sum() / group[quantity_col].sum()
    return pd.DataFrame(d, index=['weighted_avg_price'])


def aggregate_data(df, second_column, aggregation_rules, second):
    """
    聚合数据函数。
    """
    # 计算时间窗口
    df['time_window'] = (df[second_column] // second) * second

    # 初始化一个空的DataFrame用于聚合结果
    aggregated_df = pd.DataFrame()

    # 特殊处理加权平均
    if 'wavg_aggrule' in aggregation_rules:
        # 传递特定的列名到wavg函数
        temp_df = df.groupby('time_window').apply(
            lambda x: wavg(x, aggregation_rules['wavg'][0], aggregation_rules['wavg'][1])).reset_index()
        # 合并到聚合结果DataFrame
        aggregated_df = pd.merge(aggregated_df, temp_df, on='time_window',
                                 how='outer') if not aggregated_df.empty else temp_df

    # 遍历聚合规则，排除wavg后，对每个规则应用相应的聚合方法
    for column, method in {k: v for k, v in aggregation_rules.items() if k != 'wavg'}.items():
        # 应用普通聚合方法
        temp_df = df.groupby('time_window')[column].agg(method).reset_index().rename(
            columns={column: f"{method}_{column}"})
        aggregated_df = pd.merge(aggregated_df, temp_df, on='time_window',
                                 how='outer') if not aggregated_df.empty else temp_df

    return aggregated_df


#%%
# file_name = "UoB_Set01_2025-01-02LOBs.csv"
# df = pd.read_csv(file_name)

df = data_csv

#%%
# temp_df = df.groupby('time_window').apply(wavg).reset_index()
# aggregated_df = pd.merge(aggregated_df, temp_df, on='time_window', how='outer') if not aggregated_df.empty else temp_df

second_column = 'timestamp'
second = 5
df['time_window'] = (df[second_column] // second) * second

weighted_avg_per_window = df.groupby('time_window').apply(lambda x: wavg(x, 'price', 'quantity'))


# for column, method in aggregation_rules.items():
#     if method == 'wavg':
#         temp_df = df.groupby('time_window').apply(wavg).reset_index()
#         aggregated_df = pd.merge(aggregated_df, temp_df, on='time_window', how='outer') if not aggregated_df.empty else temp_df

#%%

# 定义聚合规则，例如: {'feature1': 'mean', 'feature2': 'sum'}
# 可以根据需要添加新的特征和聚合规则
aggregation_rules = {
    'avg_price': 'mean',  # 假设 avg_price 需要求平均值
    #'max_bid': 'max',  # 假设 max_bid 需要求最大值
    #使用自定义方法
    'min_ask': divide_by_2,  # 假设 feature2 需要求和，根据实际情况添加或修改
}
aggregated_df = aggregate_data(df, 'time', aggregation_rules,8)



