# 预加载模块
from pandas import Timedelta
import glob
import re
import os
import ast
import pandas as pd
from pandas import Timedelta
import glob  ## 电脑上
import datetime
## 下面三个是为了用AWS的EC2和S3
# import boto3
# from io import BytesIO
# from io import StringIO  ## 为了后面存文件

#%%
## 读入文件 read the files
def read_csvfile(file_path):
    file_name = os.path.basename(file_path)
    date = file_name.split('_')[2][:10]
    column_names = ['timestamp', 'price', 'quantity']
    data = pd.read_csv(file_path, header=None, names=column_names, usecols=[0, 1, 2])
    data['date'] = date
    return data

#%%
## 聚合--每5秒聚合成一个时间戳
# def aggregate_data(df, second_column, aggregation_rules,second):
#     df['time_window'] = (df[second_column] // second) * second
#     agg_df = df.groupby('time_window').agg(aggregation_rules).reset_index()
#     return agg_df
# test:
# aggregation_rules = {'price': 'mean'}  ## price: the true price 真实成交价
# agg_df = aggregate_data(data_csv, second_column='timestamp', aggregation_rules=aggregation_rules, second=second)

'''
Tapes dataset:
['time_window', 'mean_price', 'max_price', 'min_price',
       'most_frequent_price', 'total_quantity', 'weighted_avg_price',
       'diff_meanAndWeightedPrice']  
       'total_quantity': 时间窗内的累计成交量
       'most_frequent_price': 成交数最多的价格————时间窗内较优价格
'''
import pandas as pd
import datetime


def aggregate_Tapes(df, second_column, second):
    # 创建一个表示时间窗口的列
    df['time_window'] = (df[second_column] // second) * second
    # 计算每个交易的加权贡献
    df['weighted_price'] = df['price'] * df['quantity']

    aggregation_rules = {
        'price': ['mean', 'max', 'min'],  # 最大价格、最小价格、平均价格
        'quantity': 'sum',  # 成交数量
        'weighted_price': 'sum'  # 加权价格总和
    }
    agg_df = df.groupby('time_window').agg(aggregation_rules).reset_index()
    agg_df['weighted_avg_price'] = agg_df[('weighted_price', 'sum')] / agg_df[('quantity', 'sum')]

    # 处理众数，注意确保处理空值
    mode_prices = df.groupby('time_window')['price'].apply(lambda x: x.mode()[0] if not x.mode().empty else None)
    agg_df['most_frequent_price'] = mode_prices.values

    # 重新命名列以使输出更清晰
    agg_df.columns = ['time_window', 'mean_price', 'max_price', 'min_price', 'total_quantity',
                      'total_weighted_price', 'weighted_avg_price', 'most_frequent_price']
    agg_df['diff_meanAndWeightedPrice'] = agg_df['weighted_avg_price'] - agg_df['mean_price']

    # 删除不再需要的 total_weighted_price 列
    agg_df.drop(columns='total_weighted_price', inplace=True)

    return agg_df
#%%
######################################################################################################################

def add_quotes_to_specific_word(s, word):
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern, f"'{word}'", s)

## list.insert(position=2, data='new data')
def insert_date(list, position, date):
    # date = pd.to_datetime(file_name[10:20])
    # date = file_name[10:20]
    # print(type(list))
    list.insert(position, date)
    return list

## ----------------------------------
def read_LOBtxt(file_path):
    LOB_list = []  # 用于存储
    file_name = os.path.basename(file_path)
    # column_names = ['timestamp', 'price', 'quantity']
    with open(file_path, 'r') as file:
        txt = file.read()
        data = add_quotes_to_specific_word(txt, 'Exch0')
        lines = data.split('\n')
        for line in lines:
            if line:
                each_line = list(ast.literal_eval(line))  # 转换行
                date = file_name[10:20]
                line_withdate = insert_date(each_line, 3, date)  # 假设在列表的末尾插入日期
                LOB_list.append(line_withdate)
    return LOB_list, file_name


#%%
## split_txtdata from somethingaboutlist.py split_txtdata_4()
# 第一列 时间戳，第二列 bid订单[ [价格, 数量],[价格, 数量],...,[价格，数量] ]，第三列 ask定价单[[价格, 数量],...,[价格, 数量]]
# 第四列 max_bid价格 第五列 min_ask价格
def split_txtdata(dataset:list):
    timestamp = []
    bid = []
    max_bid = []
    ask = []
    min_ask = []
    # 提取时间戳
    for i in range(len(dataset)):
        timestamp.append(dataset[i][0])  ## 时间戳
    # 提取 价格单
    for i in range(len(dataset)):
        # 价格单中 买家给出的价格单 a buy list，只记录价格和数量
        bid.extend([dataset[i][2][0][1]])
        # 价格单中 卖家给出的价格单 a sell list
        ask.extend([dataset[i][2][1][1]])

    # ## 找到max_bid： bid订单在当前时间戳的最大价格
    # for row in dataset:
    #     max_bid_price = []  ## 保存bid最高价
    #     bid_prices_row = []  ## 保存每行的价格序列
    #     if not len(row[2][0][1]):
    #         max_bid.extend([0])
    #         continue
    #     for i in range(len(row[2][0][1])):
    #         bid_prices_row.append(row[2][0][1][i][0])
    #     max_bid_price = [max(bid_prices_row)]
    #     max_bid.extend(max_bid_price)
    # ## 找到min_ask: ask订单在当前时间戳的最低价格
    # for row in dataset:
    #     min_ask_price = []
    #     ask_prices_row = []
    #     if not len(row[2][1][1]):
    #         min_ask.extend([0])
    #         continue
    #     for i in range(len(row[2][1][1])):
    #         ask_prices_row.append(row[2][1][1][i][0])
    #     min_ask_price = [min(ask_prices_row)]
    #     min_ask.extend(min_ask_price)

    dataset = {'timestamp': timestamp, 'bid': bid, 'ask': ask}
    # dataset = {'timestamp': timestamp, 'bid': bid, 'ask': ask, 'max_bid': max_bid, 'min_ask': min_ask}
    print('The type of dataset is', type(dataset))
    print('The names of the keys are : ', dataset.keys())
    return dataset

##########################################################################################################
##########################################################################################################
#%%
# ## clean the data ## 清除异常点/离群点
# # 根据分布
# # 提取所有 bid 和 ask 中的价格，并分别处理
# def extract_prices(rows):
#     prices = []
#     for row in rows:
#         prices.extend([price[0] for price in row])
#     return prices
#
# bid_prices = extract_prices(df['bid'])
# ask_prices = extract_prices(df['ask'])
#
# # 分别计算 bid 和 ask 的分位数
# bid_lower_quantile = pd.Series(bid_prices).quantile(0.01) if bid_prices else None
# bid_upper_quantile = pd.Series(bid_prices).quantile(0.99) if bid_prices else None
# ask_lower_quantile = pd.Series(ask_prices).quantile(0.01) if ask_prices else None
# ask_upper_quantile = pd.Series(ask_prices).quantile(0.99) if ask_prices else None
# # 函数以清除离群点
# def remove_outliers(row, lower_quantile, upper_quantile):
#     # 创建一个新列表来存储不是离群点的订单
#     filtered_orders = []
#     for price, quantity in row:
#         if lower_quantile <= price <= upper_quantile:
#             filtered_orders.append([price, quantity])
#     return filtered_orders
#
# # 应用函数移除离群点
# df['bid'] = df['bid'].apply(lambda x: remove_outliers(x, bid_lower_quantile, bid_upper_quantile))
# df['ask'] = df['ask'].apply(lambda x: remove_outliers(x, ask_lower_quantile, ask_upper_quantile))


#%%
## find max_bid and min_ask in Dataset(df, cleaned)
def find_max_min_price(dataset):
    max_bid = []
    min_ask = []
    timestamp = dataset['timestamp']
    ## 找到max_bid： bid订单在当前时间戳的最大价格
    for row in dataset['bid']:   ## row 每一行bid订单列表
        max_bid_price = []  ## 保存bid最高价
        bid_prices_row = []  ## 保存每行的价格序列
        if not len(row):
            max_bid.extend([0])
            continue
        for i in range(len(row)):
            bid_prices_row.append(row[i][0])
        max_bid_price = [max(bid_prices_row)]
        max_bid.extend(max_bid_price)

    ## 找到min_ask: ask订单在当前时间戳的最低价格
    for row in dataset['ask']:
        min_ask_price = []
        ask_prices_row = []
        if not len(row):
            min_ask.extend([0])
            continue
        for i in range(len(row)):
            ask_prices_row.append(row[i][0])
        min_ask_price = [min(ask_prices_row)]
        min_ask.extend(min_ask_price)
    new_dataset = {'timestamp': timestamp, 'max_bid': max_bid, 'min_ask': min_ask}
    new_df = pd.DataFrame(new_dataset)
    ## 使用Pandas的 merge 方法合并两个DataFrame
    df = pd.merge(dataset, new_df, on='timestamp', how='outer')

    return df


# find_max_min_prices（）计算最值
# def find_max_min_prices(df):
#     # 初始化价格列表
#     bid_prices = []
#     ask_prices = []
#
#     # 遍历 DataFrame 中的每一行来提取价格
#     for index, row in df.iterrows():
#         if row['bid']:
#             bid_prices.extend([bid[0] for bid in row['bid']])
#         if row['ask']:
#             ask_prices.extend([ask[0] for ask in row['ask']])
#
#     # 计算最大值和最小值
#     max_bid_price = max(bid_prices) if bid_prices else None
#     min_bid_price = min(bid_prices) if bid_prices else None
#     max_ask_price = max(ask_prices) if ask_prices else None
#     min_ask_price = min(ask_prices) if ask_prices else None
#     return max_bid_price, min_bid_price, max_ask_price, min_ask_price
##########################################################################################################
##########################################################################################################
#%%

def before_agg_get_feature(df):   ## bid_price 和 ask_price 是bid和ask的订单(list)  ## 是split_txtdata()中的 bid和ask列
    df['bid_cumulative_depth'] = df['bid'].apply(lambda x: sum([i[1] for i in x]) if x else None)
    df['ask_cumulative_depth'] = df['ask'].apply(lambda x: sum([i[1] for i in x]) if x else None)
    df['avg_weight_bid'] = df['bid'].apply(
        lambda x: sum([i[0] * i[1] for i in x]) / sum([i[1] for i in x]) if x else None)
    df['avg_weight_ask'] = df['ask'].apply(
        lambda x: sum([i[0] * i[1] for i in x]) / sum([i[1] for i in x]) if x else None)
    #
    # aggregation_rules = {
    #     'price': ['max', 'min', pd.Series.mode],  # 最大价格、最小价格
    #
    # }
    #
    # # 对数据进行分组并应用聚合规则
    # agg_df = df.groupby('time_window').agg(aggregation_rules).reset_index()
    return df


def aggregate_data(df, second_column:'timestamp', aggregation_rules,second):
    df['time_window'] = (df[second_column] // second) * second
    aggregated_df = df.groupby('time_window').agg(aggregation_rules).reset_index() # 根据时间窗口聚合数据
    return aggregated_df

#%%
def after_agg_get_feature(df):
    df['avg_price'] = (df['max_bid'] + df['min_ask']) / 2
    df['avg_price_change'] = df['avg_price'] / df['avg_price'].shift(1) - 1
    df['bid_level_diff'] = df['max_bid'] / df['avg_price'] - 1
    df['ask_level_diff'] = df['min_ask'] / df['avg_price'] - 1
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
##########################################################################################################
##########################################################################################################
## 运行代码
##########################################################################################################
##########################################################################################################
## Tapes files
# 指定文件夹路径
# input_path_Tapes = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry'
input_path_Tapes = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/Tapes'

output_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/whole_dataset_processed'
# 获取文件夹内所有特定类型的文件，
file_paths_Tapes = glob.glob(os.path.join(input_path_Tapes, '*.csv'))  # 根据需要修改文件类型

total_Tapes = pd.DataFrame()
n=0
second = 30
date = []
# 遍历文件路径列表，处理每个文件
for file_path in file_paths_Tapes:
    original_file_name = os.path.basename(file_path)
    print("--------------Processing Tapes files:", original_file_name,"--------------")
    data_Tapes= read_csvfile(file_path)
    agg_Tapes = aggregate_Tapes(data_Tapes, second_column='timestamp', second=second)
    date = pd.to_datetime(original_file_name[10:-8])
    # date = original_file_name[0][-19:-9]
    agg_Tapes['time'] = agg_Tapes['time_window'].apply(lambda x: date + datetime.timedelta(seconds=x))
    agg_Tapes['time'] += Timedelta(hours=8)  # 需要加载包 from pandas import Timedelta
    agg_Tapes.insert(0, 'time', agg_Tapes.pop('time'))
    agg_Tapes['date'] = date
    total_Tapes = pd.concat([total_Tapes, agg_Tapes], ignore_index=True)
#%%

new_file_name = 'total_Tapes_30s.csv'
# 3. 构建新的文件路径
output_file_path = os.path.join(output_path, new_file_name)
# 处理完毕后的DataFrame可以进行保存或其他操作
# 例如，保存到CSV文件
total_Tapes.to_csv(output_file_path, index=False)
print("--------------All Tapes files processed successfully!----------------")

#%%
####################################################################
#####################################################################
## LOB files
# 指定文件夹路径
input_path_LOB = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOBs'
# input_path_LOB = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry_lob'

# output_path ## 同上面的
# 获取文件夹内所有特定类型的文件，比如.txt文件
file_paths_LOB = glob.glob(os.path.join(input_path_LOB, '*.txt'))  # 根据需要修改文件类型


aggregation_rules = {
    'max_bid': 'mean',  # 假设 max_bid 需要求最大
    'min_ask': 'mean',  # 假设 min_ask 需要求平均值
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
# 遍历文件路径列表，处理每个文件
total_LOB = pd.DataFrame()
n = 0
second = 30
# second ## 前面已经设定过了
date = []

for file_path in file_paths_LOB:
    original_file_name = os.path.basename(file_path)  ## 从完整的文件路径中提取文件名。'UoB_Set01_2025-01-02LOBs.txt'
    print("--------------Processing LOB files:", original_file_name,"--------------")
    LOB_list, file_names = read_LOBtxt(file_path)
    dataset_bidask = split_txtdata(LOB_list)
    df = pd.DataFrame(dataset_bidask)
    def extract_prices(rows):  ## 提取所有 bid 和 ask 中的价格，并分别处理
        prices = []
        for row in rows:
            prices.extend([price[0] for price in row])
        return prices
    bid_prices = extract_prices(df['bid'])   ##??? 为什么标红
    ask_prices = extract_prices(df['ask'])
    ## 分别计算 bid 和 ask 的分位数
    bid_lower_quantile = pd.Series(bid_prices).quantile(0.01) if bid_prices else None
    bid_upper_quantile = pd.Series(bid_prices).quantile(0.99) if bid_prices else None
    ask_lower_quantile = pd.Series(ask_prices).quantile(0.01) if ask_prices else None
    ask_upper_quantile = pd.Series(ask_prices).quantile(0.99) if ask_prices else None
    def remove_outliers(row, lower_quantile, upper_quantile):      ## 函数以清除离群点
        # 创建一个新列表来存储不是离群点的订单
        filtered_orders = []
        for price, quantity in row:
            if lower_quantile <= price <= upper_quantile:
                filtered_orders.append([price, quantity])
        return filtered_orders
    # 应用函数移除离群点
    df['bid'] = df['bid'].apply(lambda x: remove_outliers(x, bid_lower_quantile, bid_upper_quantile))
    df['ask'] = df['ask'].apply(lambda x: remove_outliers(x, ask_lower_quantile, ask_upper_quantile))

    df = find_max_min_price(df)
    df = before_agg_get_feature(df)
    df = aggregate_data(df, 'timestamp', aggregation_rules, second)
    date = pd.to_datetime(original_file_name[10:-8])    ## date = file_names[2][-18:-8]
    df['time'] = df['time_window'].apply(lambda x: date + datetime.timedelta(seconds=x))
    df['time'] += Timedelta(hours=8)  # 需要加载包 from pandas import Timedelta
    df.insert(0, 'time', df.pop('time'))
    df['date'] = date
    total_LOB = pd.concat([total_LOB, df], ignore_index=True)

#%%
total_LOB = after_agg_get_feature(total_LOB)
# total_LOB_copy = total_LOB.copy()
total_LOB = mark_label(total_LOB,10,0.1)  ## 需要调参数

new_file_name = 'total_lob_30s_k10.csv'
# 3. 构建新的文件路径
output_file_path = os.path.join(output_path, new_file_name)
# 处理完毕后的DataFrame可以进行保存或其他操作
# 例如，保存到CSV文件
total_LOB.to_csv(output_file_path, index=False)

# total_LOB_copy.to_csv(output_file_path, index=False)  ## 没有标签的数据
print("--------------All LOB files processed successfully!----------------")

# #%%
# whole_dataset = pd.merge(total_LOB, total_Tapes, on='time', how='outer')
#
# new_file_name = 'whole_dataset_60s.csv'
# output_file_path = os.path.join(output_path, new_file_name)
# whole_dataset.to_csv(output_file_path, index=False)
# print("--------------Two datasets processed successfully!----------------")

