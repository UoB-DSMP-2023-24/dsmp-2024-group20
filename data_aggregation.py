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

    return aggregated_df



#%%
file_name = "UoB_Set01_2025-01-02LOBs.csv"
df = pd.read_csv(file_name)
# 定义聚合规则，例如: {'feature1': 'mean', 'feature2': 'sum'}
# 可以根据需要添加新的特征和聚合规则
aggregation_rules = {
    'avg_price': 'mean',  # 假设 avg_price 需要求平均值
    #'max_bid': 'max',  # 假设 max_bid 需要求最大值

}
aggregated_df = aggregate_data(df, 'time', aggregation_rules,5)



