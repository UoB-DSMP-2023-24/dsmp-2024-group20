import boto3
import pandas as pd
from pandas import Timedelta
import glob
import re
import os
import ast
import pandas as pd
from pandas import Timedelta
import glob  ## 电脑上
import datetime
import pandas as pd
import datetime
import boto3
import pandas as pd
from io import BytesIO
from io import StringIO  ## 为了后面存文件
#%%
# 定义处理CSV文件的函数
## 读入文件 read the files
def read_csvfile(bucket, key):
    print("-----")
    print("bucket")
    print(bucket)
    print("key")
    print(key)
    obj = bucket.Object(key)
    print(object)
    print(obj)
    data = pd.read_csv(obj.get()['Body'])
    print(data)
    file_name = os.path.basename(object.key)
    date = file_name.split('_')[2][:10]
    column_names = ['timestamp', 'price', 'quantity']
    data = pd.read_csv(obj.get()['Body'], header=None, names=column_names, usecols=[0, 1, 2])
    data['date'] = date
    return data

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

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
#%%

def add_quotes_to_specific_word(s, word):
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern, f"'{word}'", s)

## list.insert(position=2, data='new data')
def insert_date(list, position, date):
    # date = pd.to_datetime(file_name[10:20])
    # date = file_name[10:20]
    list.insert(position, date)
    return list

# 定义处理TXT文件的函数
# def process_txt_file(bucket, key):
#     obj = bucket.Object(key)
#     data = obj.get()['Body'].read().decode('utf-8')
#     # 对数据进行处理
#     print(data)

def read_LOBtxt(bucket, key):
    LOB_list = []  # 用于存储
    obj = bucket.Object(key)
    txt = obj.get()['Body'].read().decode('utf-8')
    file_name = os.path.basename(object.key)
    # column_names = ['timestamp', 'price', 'quantity']
    data = add_quotes_to_specific_word(txt, 'Exch0')
    lines = data.split('\n')
    for line in lines:
        if line:
            each_line = ast.literal_eval(line)  # 转换行
            date = file_name[10:20]
            line_withdate = insert_date(each_line, 3, date)  # 假设在列表的末尾插入日期
            LOB_list.append(line_withdate)
    return LOB_list

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
    dataset = {'timestamp': timestamp, 'bid': bid, 'ask': ask}
    # dataset = {'timestamp': timestamp, 'bid': bid, 'ask': ask, 'max_bid': max_bid, 'min_ask': min_ask}
    print('The type of dataset is', type(dataset))
    print('The names of the keys are : ', dataset.keys())
    return dataset

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

def before_agg_get_feature(df):   ## bid_price 和 ask_price 是bid和ask的订单(list)  ## 是split_txtdata()中的 bid和ask列
    df['bid_cumulative_depth'] = df['bid'].apply(lambda x: sum([i[1] for i in x]) if x else None)
    df['ask_cumulative_depth'] = df['ask'].apply(lambda x: sum([i[1] for i in x]) if x else None)
    df['avg_weight_bid'] = df['bid'].apply(
        lambda x: sum([i[0] * i[1] for i in x]) / sum([i[1] for i in x]) if x else None)
    df['avg_weight_ask'] = df['ask'].apply(
        lambda x: sum([i[0] * i[1] for i in x]) / sum([i[1] for i in x]) if x else None)

    return df


def aggregate_data(df, second_column:'timestamp', aggregation_rules,second):
    df['time_window'] = (df[second_column] // second) * second
    aggregated_df = df.groupby('time_window').agg(aggregation_rules).reset_index() # 根据时间窗口聚合数据
    return aggregated_df

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
######################################################################################################
######################################################################################################
######################################################################################################
# 连接到S3
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket_name = 'lob-data-processed'
bucket = s3.Bucket(bucket_name)



# 处理Tapes目录下的CSV文件
total_Tapes = pd.DataFrame()
n=0
second = 300
date = []
# 遍历文件路径列表，处理每个文件


tapes_prefix = 'raw-dataset/Tapes/'
for obj in bucket.objects.filter(Prefix=tapes_prefix):
    print("obj")
    print(obj)
    if obj.key.endswith('.csv'):
        original_file_name = os.path.basename(obj.key)
        print("--------------Processing Tapes files:", original_file_name,"--------------")
        data_Tapes= read_LOBtxt(bucket, obj.key)
        agg_Tapes = aggregate_Tapes(data_Tapes, second_column='timestamp', second=second)
        date = pd.to_datetime(original_file_name[10:-8])
        # date = original_file_name[0][-19:-9]
        agg_Tapes['time'] = agg_Tapes['time_window'].apply(lambda x: date + datetime.timedelta(seconds=x))
        agg_Tapes['time'] += Timedelta(hours=8)  # 需要加载包 from pandas import Timedelta
        agg_Tapes.insert(0, 'time', agg_Tapes.pop('time'))
        agg_Tapes['date'] = date
        total_Tapes = pd.concat([total_Tapes, agg_Tapes], ignore_index=True)

# new_file_name = 'total_Tapes.csv'
# # 3. 构建新的文件路径
# output_file_path = os.path.join(output_path, new_file_name)
# # 处理完毕后的DataFrame可以进行保存或其他操作
# # 例如，保存到CSV文件
# total_Tapes.to_csv(output_file_path, index=False)
print("--------------All Tapes files processed successfully!----------------")



######################################################################################################
######################################################################################################
######################################################################################################

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
# %%
# 遍历文件路径列表，处理每个文件
total_LOB = pd.DataFrame()
n = 0
# second ## 前面已经设定过了
date = []


def extract_prices(rows):  ## 提取所有 bid 和 ask 中的价格，并分别处理
    prices = []
    for row in rows:
        prices.extend([price[0] for price in row])
    return prices


def remove_outliers(row, lower_quantile, upper_quantile):  ## 函数以清除离群点
    # 创建一个新列表来存储不是离群点的订单
    filtered_orders = []
    for price, quantity in row:
        if lower_quantile <= price <= upper_quantile:
            filtered_orders.append([price, quantity])
    return filtered_orders


# 处理LOB目录下的TXT文件
lob_prefix = 'raw-dataset/LOB/'
for obj in bucket.objects.filter(Prefix=lob_prefix):
    if obj.key.endswith('.txt'):
        original_file_name = os.path.basename(obj.key)  ## 从完整的文件路径中提取文件名。'UoB_Set01_2025-01-02LOBs.txt'
        print("--------------Processing LOB files:", original_file_name, "--------------")
        LOB_list, file_names = read_LOBtxt(bucket, obj.key)
        dataset_bidask = split_txtdata(LOB_list)
        df = pd.DataFrame(dataset_bidask)

        bid_prices = extract_prices(df['bid'])  ##??? 为什么标红
        ask_prices = extract_prices(df['ask'])
        ## 分别计算 bid 和 ask 的分位数
        bid_lower_quantile = pd.Series(bid_prices).quantile(0.01) if bid_prices else None
        bid_upper_quantile = pd.Series(bid_prices).quantile(0.99) if bid_prices else None
        ask_lower_quantile = pd.Series(ask_prices).quantile(0.01) if ask_prices else None
        ask_upper_quantile = pd.Series(ask_prices).quantile(0.99) if ask_prices else None
        # 应用函数移除离群点
        df['bid'] = df['bid'].apply(lambda x: remove_outliers(x, bid_lower_quantile, bid_upper_quantile))
        df['ask'] = df['ask'].apply(lambda x: remove_outliers(x, ask_lower_quantile, ask_upper_quantile))

        df = find_max_min_price(df)
        df = before_agg_get_feature(df)
        df = aggregate_data(df, 'timestamp', aggregation_rules, second)
        date = pd.to_datetime(original_file_name[10:-8])  ## date = file_names[2][-18:-8]
        df['time'] = df['time_window'].apply(lambda x: date + datetime.timedelta(seconds=x))
        df['time'] += Timedelta(hours=8)  # 需要加载包 from pandas import Timedelta
        df.insert(0, 'time', df.pop('time'))
        df['date'] = date
        total_LOB = pd.concat([total_LOB, df], ignore_index=True)

# %%
total_LOB = after_agg_get_feature(total_LOB)
# new_file_name = 'total_lob.csv'
print("--------------All LOB files processed successfully!----------------")

# total_LOB = mark_label(total_LOB,20,0.001)  ## 不要了

# # 3. 构建新的文件路径
# output_file_path = os.path.join(output_path, new_file_name)
# # 处理完毕后的DataFrame可以进行保存或其他操作
# # 例如，保存到CSV文件
# total_LOB.to_csv(output_file_path, index=False)

#%%
whole_dataset = pd.merge(total_LOB, total_Tapes, on='time', how='outer')
new_file_name = 'whole_dataset.csv'
# output_file_path = os.path.join(output_path, new_file_name)
# whole_dataset.to_csv(output_file_path, index=False)
print("--------------Two datasets processed successfully!----------------")

######################################################################################################
######################################################################################################
######################################################################################################
# 输出为csv文件，便于后续分析
## 指定上传的文件名
object_key = 'whole-dataset/30s/whole_dataset_30s.csv'

# 把DataFrame保存到一个字符串缓冲区中
csv_buffer = StringIO()
whole_dataset.to_csv(csv_buffer, index=True)

# 上传到S3
s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=csv_buffer.getvalue())

print(f"CSV uploaded to S3: s3://{bucket_name}/{object_key}")
