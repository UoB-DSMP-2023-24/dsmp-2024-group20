#%%
import os
import glob

# import numpy as np
import pandas as pd
import ast
from pandas import Timedelta
import datetime

def process_lob_file(file_path):
    lob = [] # 初始化空列表 lob：用于存储处理过的每一行数据
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            lob.append(list(line.split('[[')))   # split('[[') 用于分割文本，假设原始数据中买单和卖单由 [[ 分隔
            # if i == 73500:
            #     break

    # 处理bids
    bids = [] # 初始化相关列表：bids, time, n, m_price 用于存储买单的价格、时间、数量和最大价格
    time = []
    n = []
    m_price = []
    number_of_prices = 5
    for line in lob:
        try:
            if len(line) > 2:
                dt = ast.literal_eval('[[' + list(line[2].split("]]], ['ask', "))[0] + ']]')
                bids.append(dt)
                time.append(line[0].split(',')[0][1:])
                n.append(len(dt))
                m_price.append(bids[len(bids) - 1][0][0])
        except Exception as e:
            print(f"Error processing line: {line}. Error: {e}")
    ## bidsdf：使用处理过的数据列表创建一个Pandas DataFrame
    bidsdf = pd.DataFrame({'time': time,
                           "bid_price": [i[0:number_of_prices] for i in bids],    ## bid_price: bid订单列表
                           "n_bid_prices": n,
                           "max_bid": m_price})

    # 处理asks
    asks = []
    time = []
    n = []
    m_price = []
    for line in lob:
        try:
            if len(line) > 3:
                dt = ast.literal_eval('[[' + list(line[3].split("]]]]]"))[0] + ']]')
                asks.append(dt)
                time.append(line[0].split(',')[0][1:])
                n.append(len(dt))
                m_price.append(asks[len(asks) - 1][0][0])
        except Exception as e:
            print(f"Error processing line: {line}. Error: {e}")

    asksdf = pd.DataFrame({'time': time,
                           "ask_price": [i[0:number_of_prices] for i in asks],   ## ask_price: ask订单列表
                           "n_ask_price": n,
                           "min_ask": m_price})
    ## 使用Pandas的 merge 方法合并两个DataFrame，基于时间戳，并将时间转换为浮点型
    df = pd.merge(bidsdf, asksdf, on='time', how='outer').dropna()
    df['time'] = df['time'].astype(float)
    return df

def aggregate_data(df, second_column, aggregation_rules,second):
    df['time_window'] = (df[second_column] // second) * second

    # 根据时间窗口聚合数据
    aggregated_df = df.groupby('time_window').agg(aggregation_rules).reset_index()

    return aggregated_df

def before_agg_get_feature(df):

    df['bid_cumulative_depth'] = df['bid_price'].apply(lambda x: sum([i[1] for i in x]) if x else None)
    df['ask_cumulative_depth'] = df['ask_price'].apply(lambda x: sum([i[1] for i in x]) if x else None)

    df['avg_weight_bid'] = df['bid_price'].apply(
        lambda x: sum([i[0] * i[1] for i in x]) / sum([i[1] for i in x]) if x else None)
    df['avg_weight_ask'] = df['ask_price'].apply(
        lambda x: sum([i[0] * i[1] for i in x]) / sum([i[1] for i in x]) if x else None)
    return df

def after_agg_get_feature(df):
    df['avg_price'] = (df['avg_weight_bid'] + df['avg_weight_ask']) / 2
    df['avg_price_change'] = df['avg_price'] / df['avg_price'].shift(1) - 1
    df['bid_level_diff'] = df['avg_weight_bid'] / df['avg_price'] - 1
    df['ask_level_diff'] = df['avg_weight_ask'] / df['avg_price'] - 1
    df['bid_ask_depth_diff'] = ((df['bid_cumulative_depth'] - df['ask_cumulative_depth'])/
                                (df['bid_cumulative_depth'] + df['ask_cumulative_depth']))
    df['next_avg_price_change'] = (df['avg_price'].shift(-1)-df['avg_price'])/df['avg_price']

    return df
def mark_label(df,k,thresholds):
    #0:stay,1:up，2:down
    # df['m_minus'] = df['avg_price'].rolling(window=k).mean()
    df['m_plus'] = df['avg_price'].shift(-1).rolling(window=k, min_periods=1).mean().shift(-k+1)
    df['l_t'] = (df['m_plus'] - df['avg_price']) / df['avg_price']
    df['label'] = 0
    df.loc[df['l_t'] > thresholds, 'label'] = 1
    df.loc[df['l_t'] < -thresholds, 'label'] = 2

    return df

#%%
# 指定文件夹路径
input_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOBs'
output_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOBdata_process_weight/avg/300s'
# 获取文件夹内所有特定类型的文件，比如.txt文件
file_paths = glob.glob(os.path.join(input_path, '*.txt'))  # 根据需要修改文件类型
    ## file_paths[i][-18:-8]：第i个文件的年月日，例如“2025-01-02”。

aggregation_rules = {
    'avg_weight_bid': 'mean',  # 假设 max_bid 需要求最大
    'avg_weight_ask': 'mean',  # 假设 min_ask 需要求平均值
    # 'bid_ask_depth_diff': 'mean',  # 假设 bid_ask_depth_diff 需要求平均值

    # 'next_avg_price_change': 'mean',
    # 'avg_price': 'mean',  # 假设 avg_price 需要求平均值
    # 'avg_price_change': 'mean',  # 假设 mid_price_change 需要求平均值
    # 'bid_level_diff': 'mean',  # 假设 bid_level_diff 需要求平均值
    # 'ask_level_diff': 'mean',  # 假设 ask_level_diff 需要求平均值

    'bid_cumulative_depth': 'mean',  # 假设 bid_cumulative_depth 需要求平均值
    'ask_cumulative_depth': 'mean',  # 假设 ask_cumulative_depth 需要求平均值


}

#%%
######## 标签有问题：每日的最后20个标签不正确
# 遍历文件路径列表，处理每个文件
date = []
for file_path in file_paths:
    original_file_name = os.path.basename(file_path)  ## 从完整的文件路径中提取文件名。'UoB_Set01_2025-01-02LOBs.txt'
    print("--------------Processing file:", original_file_name,"--------------")
    df = process_lob_file(file_path) # 读文件， txt -> df
    df = before_agg_get_feature(df) # 在数据聚合之前，该函数计算一些基本的统计特征，如累计深度和加权平均价格
    df = aggregate_data(df, 'time', aggregation_rules,300)
    df = after_agg_get_feature(df) # 在数据聚合之后，该函数计算更多的特征，包括价格变化、买卖差价深度差异等。
    df = mark_label(df,20,0.0001) # 根据未来价格变动，为数据打标签（0：保持不变，1：上升，2：下降）。
    date = pd.to_datetime(original_file_name[10:-8])
    df['time'] = df['time_window'].apply(lambda x: date + datetime.timedelta(seconds=x))
    df['time'] += Timedelta(hours=8)  # 需要加载包 from pandas import Timedelta
    df.insert(0, 'time', df.pop('time'))
    df['date'] = date

    new_file_name = os.path.splitext(original_file_name)[0] + '.csv' # 新文件的文件名，存储成csv格式。
    # 3. 构建新的文件路径
    output_file_path = os.path.join(output_path, new_file_name)
    # 处理完毕后的DataFrame可以进行保存或其他操作
    # 例如，保存到CSV文件
    df.to_csv(output_file_path, index=False)
    # if True:
    #     break
print("--------------All files processed successfully!----------------")


#%%
## 所有数据放到一起 # 标签正确
## 不需要合并
total_df = pd.DataFrame()
n=0
# 遍历文件路径列表，处理每个文件
for file_path in file_paths:
    original_file_name = os.path.basename(file_path)
    print("--------------Processing file:", original_file_name,"--------------")
    df = process_lob_file(file_path)
    df = before_agg_get_feature(df)
    df = aggregate_data(df, 'time', aggregation_rules,300)
    date_part = original_file_name.split('_')[2]  # 文件名格式: "UoB_Set01_2025-01-02LOBs.csv"
    date = date_part[:10]
    df['date'] = date
    df.insert(0, 'date', df.pop('date'))
    total_df = pd.concat([total_df, df], ignore_index=True)
    # n+=1
    # if n == 2:
    #     break

total_df = after_agg_get_feature(total_df)
total_df = mark_label(total_df,20,0.001)


new_file_name = 'total_lob.csv'
# 3. 构建新的文件路径
output_file_path = os.path.join(output_path, new_file_name)
# 处理完毕后的DataFrame可以进行保存或其他操作
# 例如，保存到CSV文件
total_df.to_csv(output_file_path, index=False)

print("--------------All files processed successfully!----------------")

#%%
'''
import pandas as pd
import glob
import os
## output_path 是原文件(每天的LOB数据的csv格式文件)所在的文件夹地址
'''
## 每20个原文件合并成一个csv文件
directory_path = output_path
## 所有原文件的目录
file_paths = glob.glob(os.path.join(directory_path, '*.csv'))

## 选择原文件中需要的列
# columns_to_keep = ['time', 'avg_weight_bid', 'avg_weight_ask', 'next_avg_price_change']

## output_directory:保存新csv文件(combined_df文件)的文件夹地址.
output_directory = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOBdata_process_weight/300s/combined_df_agg300s'

for i in range(0, len(file_paths), 20):
    batch_files = file_paths[i:i+20]  # 获取当前批次的文件路径列表
    dfs = [pd.read_csv(file) for file in batch_files]  # 读取每个文件为DataFrame
#    dfs = [pd.read_csv(file, usecols=columns_to_keep) for file in batch_files] ## 只读取某些列的数据
    combined_df = pd.concat(dfs, ignore_index=True)  # 合并当前批次的DataFrame
    ## 保存合并后的文件，文件名“combined_1.csv” 第一批合并的文件
    name = f'combined_{i//20 + 1}.csv' # 文件名
    combined_df_path = os.path.join(output_directory, name)
    combined_df.to_csv(combined_df_path, index=False)

#%%
## 原csv文件全放到一起
dfs = [pd.read_csv(file) for file in file_paths]  # 读取每个文件为DataFrame
# dfs = [pd.read_csv(file, usecols=columns_to_keep) for file in batch_files] ## 只读取某些列的数据
combined_df = pd.concat(dfs, ignore_index=True)  # 合并当前批次的DataFrame

## 保存合并后的文件
name = 'combined_df.csv'
combined_df_path = os.path.join(output_directory, name)
combined_df.to_csv(combined_df_path, index=False)