import re
import os
import ast
import numpy as np
import pandas as pd
#%%
def add_quotes_to_specific_word(s, word):
    # 使用正则表达式为特定单词添加引号
    # \b 是单词边界，确保整个单词被匹配
    # 使用 re.escape 来处理 word 中可能包含的任何特殊字符
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern, f"'{word}'", s)

# # 测试字符串
# test_str = "[0.000, Exch0, [['bid', []], ['ask', []]]]"
# # 转换字符串，只为"Exch0"添加引号
# corrected_str = add_quotes_to_specific_word(test_str, 'Exch0')
# print(corrected_str)

## list.insert(position=2, data='new data')

test_list = [0.0, 'Exch0', [['bid', []], ['ask', []]]]

def insert_date(list, position, date):
    # date = pd.to_datetime(file_name[10:20])
    # date = file_name[10:20]
    list.insert(position, date)
    return list

# file_names[0][10:14] ## '2025'
# file_names[0][15:17] ## '01'
# file_names[0][18:20] ## '02'

# 文件路径
file_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry_lob/'

## ----------------------------------
def read_LOBtxt(directory):
    LOB_list = []  # 用于存储
    file_names = os.listdir(directory)  # 获取目录中的所有文件名  # import os
    # column_names = ['timestamp', 'price', 'quantity']
    for file_name in file_names:
        if file_name.endswith('.txt'):  # 先检查是否是 txt 文件
            file_path = os.path.join(directory, file_name)  # 正确地构建文件路径
            with open(file_path, 'r') as file:
                txt = file.read()
                data = add_quotes_to_specific_word(txt, 'Exch0')
                lines = data.split('\n')
                # LOB_list = [ast.literal_eval(line) for line in lines if line]  # import ast
                for line in lines:
                    if line:
                        each_line = ast.literal_eval(line)  # 转换行
                        # 在转换后的列表中插入日期信息
                        date = file_name[10:20]
                        line_withdate = insert_date(each_line, 3, date)  # 假设在列表的末尾插入日期
                        LOB_list.append(line_withdate)
    return LOB_list, file_names

### ????
# for line in lines:
#     if line:
#         LOB_list = LOB_list.append(ast.literal_eval(line))

#%%
# LOB_list, file_names_lob = read_LOBtxt(file_path)
LOB_list3, file_names3 = read_LOBtxt(file_path)

import csv
with open("LOB_list2.csv", "w", newline='') as file:
    # 创建一个 CSV writer 对象
    csv_writer = csv.writer(file)
    # 写入列表中的数据行
    csv_writer.writerows(LOB_list3)

# df_bid.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset5_bid.csv', index=False)

## ----------------------------------
#%%
## ---------------------------------------------
###  this function is "split_txtdata_5(dataset:list)" in somethingaboutlist.py
def split_txtdata(dataset:list, filenames):
    date = []
    date_len = 0
    timestamp_bid = []
    timestamp_ask = []
    bid_price = []
    bid_quantity = []
    bid = []
    ask_price = []
    ask_quantity = []
    ask = []
    for i in range(len(dataset)):   ## 按每条list数据处理
        for j in range(len(dataset[i][2][0][1])):
            timestamp_bid.append(dataset[i][0])
            bid.append(dataset[i][2][0][1][j])
            bid_price.append(dataset[i][2][0][1][j][0])
            bid_quantity.append(dataset[i][2][0][1][j][1])
        for q in range(len(dataset[i][2][1][1])):
            timestamp_ask.append(dataset[i][0])
            ask.append(dataset[i][2][1][1][q])
            ask_price.append(dataset[i][2][1][1][q][0])
            ask_quantity.append(dataset[i][2][1][1][q][1])
    for filename in filenames:
        if i in range (len(dataset)) and dataset[i][3] == filename:
            date_len += 1
        date_ = [filename[10:20]] * date_len
        date_.extend(date_)

    dataset = {"date": date, 'bid': bid, 'timestamp_bid': timestamp_bid, 'bid_price': bid_price, 'bid_quantity': bid_quantity, 'ask': ask, 'timestamp_ask': timestamp_ask, 'ask_price': ask_price, 'ask_quantity': ask_quantity}
    print('The type of dataset is', type(dataset))
    print('The names of the keys are : ', dataset.keys())
    return dataset

LOB_list = LOB_list2
dataset_bidask = split_txtdata(LOB_list)

#%%
## 将dataset_bidask 转化成两个 数据框

def bidaskList_to_dataframe(dataset_bidask):
    keys_bid = ['timestamp_bid', 'bid_price', 'bid_quantity']
    keys_ask = ['timestamp_ask', 'ask_price', 'ask_quantity']
    ask_dict = {key: dataset_bidask[key] for key in keys_ask if key in dataset_bidask}
    bid_dict = {key: dataset_bidask[key] for key in keys_bid if key in dataset_bidask}
    df_bid = pd.DataFrame(data=bid_dict)
    df_ask = pd.DataFrame(data=ask_dict)
    return df_bid, df_ask

# keys_bid = ['timestamp_bid','bid_price','bid_quantity']
# bid_dict = {key: dataset_bidask5[key] for key in keys_bid if key in dataset_bidask5}
# df_bid = pd.DataFrame(data=bid_dict)
#
# keys_ask = ['timestamp_ask', 'ask_price', 'ask_quantity']
# ask_dict = {key: dataset_bidask5[key] for key in keys_ask if key in dataset_bidask5}
# df_ask = pd.DataFrame(data=ask_dict)
#
# # 导出到CSV文件，这里的index=False表示不导出行索引
# df_bid.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset5_bid.csv', index=False)
# df_ask.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset5_ask.csv', index=False)


#%%
df_bid, df_ask = bidaskList_to_dataframe(dataset_bidask)

#%%
