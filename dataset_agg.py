'''
## output_path 是原文件(每天的LOB数据的csv格式文件)所在的文件夹地址
    output_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOBdata_process_weight/30s'
'''
import pandas as pd
import glob
import os

directory_path = '……'
## 所有原文件的目录
file_paths = glob.glob(os.path.join(directory_path, '*.csv'))
## output_directory:保存新csv文件(combined_df文件)的文件夹地址.
output_directory = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOBdata_process_weight/30s/combined_df_agg30s'

## 选择原文件中需要的列
# columns_to_keep = ['time', 'avg_weight_bid', 'avg_weight_ask', 'next_avg_price_change']

#%%
## 每20个原文件合并成一个新的csv文件
for i in range(0, len(file_paths), 20):
    batch_files = file_paths[i:i+20]  # 获取当前批次的文件路径列表
    dfs = [pd.read_csv(file) for file in batch_files]  # 读取每个文件为DataFrame
    # dfs = [pd.read_csv(file, usecols=columns_to_keep) for file in batch_files] ## 只读取某些列的数据
    combined_df = pd.concat(dfs, ignore_index=True)  # 合并当前批次的DataFrame
    ## 保存合并后的文件，文件名“combined_1.csv” 第一批合并的文件
    name = f'combined_{i//20 + 1}.csv' # 文件名，例子"combined_1.csv" 第1次合并的数据集。
    combined_df_path = os.path.join(output_directory, name)
    combined_df.to_csv(combined_df_path, index=False)

#%%
## 指定 某部分原文件 合并到一起。
## 调整数字索引
batch_files = file_paths[:100]  # 获取 前100个文件 的路径列表

dfs = [pd.read_csv(file) for file in batch_files]  # 读取每个文件为DataFrame
# dfs = [pd.read_csv(file, usecols=columns_to_keep) for file in batch_files] ## 只读取某些列的数据
combined_df = pd.concat(dfs, ignore_index=True)  # 合并当前批次的DataFrame



## 保存合并后的文件
name = 'combined_df.csv'
combined_df_path = os.path.join(output_directory, name)
combined_df.to_csv(combined_df_path, index=False)
