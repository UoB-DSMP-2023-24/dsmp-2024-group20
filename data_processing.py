#%%
import os
import glob

import numpy as np
import pandas as pd
import ast
def process_lob_file(file_path):
    lob = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            lob.append(list(line.split('[[')))
            # if i == 73500:
            #     break

    # 处理bids
    bids = []
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

    bidsdf = pd.DataFrame({'time': time,
                           "bid_price": [i[0:number_of_prices] for i in bids],
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
                           "ask_price": [i[0:number_of_prices] for i in asks],
                           "n_ask_price": n,
                           "min_ask": m_price})

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

    return df

def after_agg_get_feature(df):
    df['avg_price'] = (df['max_bid'] + df['min_ask']) / 2
    df['avg_price_change'] = df['avg_price'] / df['avg_price'].shift(1) - 1
    df['next_avg_price_change'] = (df['avg_price'].shift(-1)-df['avg_price'])/df['avg_price']
    df['bid_level_diff'] = df['max_bid'] / df['avg_price'] - 1
    df['ask_level_diff'] = df['min_ask'] / df['avg_price'] - 1
    return df
def mark_label(df,k,thresholds):
    #0:stay,1:up，2:down
    df['m_minus'] = df['avg_price'].rolling(window=k).mean()
    df['m_plus'] = df['avg_price'].shift(-k).rolling(window=k).mean()
    df['l_t'] = (df['m_plus'] - df['avg_price']) / df['avg_price']
    df['label'] = 0
    df.loc[df['l_t'] > thresholds, 'label'] = 1
    df.loc[df['l_t'] < -thresholds, 'label'] = 2

    return df

#%%
# 指定文件夹路径
input_path = 'C:/桌面/learning/s2/mini/JPMorgan_Set01/LOBs'
output_path = 'process_data'
# 获取文件夹内所有特定类型的文件，比如.txt文件
file_paths = glob.glob(os.path.join(input_path, '*.txt'))  # 根据需要修改文件类型
aggregation_rules = {
    'max_bid': 'mean',  # 假设 max_bid 需要求最大
    'min_ask': 'mean',  # 假设 min_ask 需要求平均值

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
for file_path in file_paths:
    original_file_name = os.path.basename(file_path)
    print("--------------Processing file:", original_file_name,"--------------")
    df = process_lob_file(file_path)
    df = before_agg_get_feature(df)

    df = aggregate_data(df, 'time', aggregation_rules,1)
    df = after_agg_get_feature(df)
    df = mark_label(df,10,0.01)
    new_file_name = os.path.splitext(original_file_name)[0] + '.csv'
    # 3. 构建新的文件路径
    output_file_path = os.path.join(output_path, new_file_name)
    # 处理完毕后的DataFrame可以进行保存或其他操作
    # 例如，保存到CSV文件
    df.to_csv(output_file_path, index=False)
    if True:
        break
print("--------------All files processed successfully!----------------")