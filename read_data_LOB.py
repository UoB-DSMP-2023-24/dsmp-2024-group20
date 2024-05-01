import re
import os
import ast
# from collections import OrderedDict  ### 不好用

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Timedelta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#%%
def add_quotes_to_specific_word(s, word):
    # 使用正则表达式为特定单词添加引号
    # \b 是单词边界，确保整个单词被匹配
    # 使用 re.escape 来处理 word 中可能包含的任何特殊字符
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern, f"'{word}'", s)

# list.insert(position=2, data='new data')
def insert_date(list, position, date):  ## 时间--年月日
    # date = pd.to_datetime(file_name[10:20])
    # date = file_name[10:20]
    list.insert(position, date)
    return list


# 文件夹路径
# file_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry_lob/'

## ----------------------------------

"""
    读取目录中某些名字的 LOB 数据文件 并返回一个包含它们的 LOB 数据列表。

    Args:
        directory: LOB 数据文件所在的目录 需要在最后追加"/"

    Returns:
        LOB_list: 一个包含所有 LOB 数据的列表
    """


def read_LOBtxt(directory, file_names):
    LOB_list = []  # 使用列表存储每个文件的数据

    for file_name in file_names:
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as file:
                txt = file.read()
                data = add_quotes_to_specific_word(txt, 'Exch0')
                lines = data.split('\n')

                file_data = []  # 存储每个文件的数据
                for line in lines:
                    if line:
                        each_line = ast.literal_eval(line)
                        date = file_name[10:20]
                        line_withdate = insert_date(each_line, 3, date)
                        file_data.append(line_withdate)

                LOB_list.append(file_data)  # 将每个文件的数据添加到 LOB_list 中

    return LOB_list, file_names



#%%
## save the List
# LOB_list, file_names = read_LOBtxt(file_path)

#%%
###########
"""
处理单个 LOB 数据文件并返回 bid 和 ask 表的数据 (按照订单展开————会有时间戳重复出现)。

Args:
    LOB_data: 一个包含单个 LOB 数据文件数据的列表

Returns:
    bid_data: Bid 表的数据 列表
    ask_data: Ask 表的数据 列表
"""


def process_LOB_file(LOB_data):
    bid_data = []
    ask_data = []

    for each_line in LOB_data:
        timestamp = each_line[0]
        price_quantity_list_bid = each_line[2][0][1]
        price_quantity_list_ask = each_line[2][1][1]
        if each_line[2][0][0] == "bid" and len(price_quantity_list_bid) > 0:
            for price_bid, quantity_bid in price_quantity_list_bid:
                bid_data.append({"timestamp": timestamp, "order": price_quantity_list_bid, "price": price_bid, "quantity": quantity_bid})
                    # 此处可以增加 市场深度，之后根据市场深度来聚合数据
        if each_line[2][1][0] == "ask" and len(price_quantity_list_ask) > 0:
            for price_ask, quantity_ask in price_quantity_list_ask:
                ask_data.append({"timestamp": timestamp, "order": price_quantity_list_ask, "price": price_ask, "quantity": quantity_ask})

        # bid_data.extend(bid_data)
        # ask_data.extend(ask_data)

    return bid_data, ask_data




#%%
#### ------------------------------------------------
"""
处理 LOB 数据
-- 暂无： 并将其输出到两个 CSV 文件中。

Args:
    LOB_list: 一个包含所有 LOB 数据的列表
    file_names: 一个包含所有文件名列表的列表
    output_path_bid: Bid 表的输出路径
    output_path_ask: Ask 表的输出路径
    决定删除output_path_bid , output_path_ask 

Returns:
    LOB_bid: 包含所有LOB中的 bid表(数据框类型)  列表
    LOB_ask: 包含所有LOB中的 ask表(数据框类型)  列表
"""


def process_LOB_data(LOB_list, file_names):
    LOB_bid = []  # 存储 bid 表的数据
    LOB_ask = []  # 存储 ask 表的数据

    for i, LOB_data in enumerate(LOB_list):
        date = file_names[i][10:20]
        bid_data, ask_data = process_LOB_file(LOB_data)
        for j in range(len(bid_data)):
            bid_data[j]["date"] = date
        for q in range(len(ask_data)):  # 使用正确的循环范围
            ask_data[q]["date"] = date

        # df_bid = pd.DataFrame(bid_data)
        # df_ask = pd.DataFrame(ask_data)
        # df_bid.to_csv(output_path_bid, index=False)
        # df_ask.to_csv(output_path_ask, index=False)

        LOB_bid.extend([bid_data])
        LOB_ask.extend([ask_data])

        bid_data = []
        ask_data = []



    return LOB_bid, LOB_ask

# ### 测试代码
# a = [LOB_list[0][50:52], LOB_list[1][50:52]]
# date = '2000-01-01'
# c = []
# bid_data = []
# ask_data = []
# for i, aa in enumerate(a[:5]):
#     print("--------------------------")
#     bid_data, ask_data = process_LOB_file(aa)
#     for j in range(len(bid_data)):
#         bid_data[j]["date"] = date
#     for q in range(len(ask_data)):  # 使用正确的循环范围
#         ask_data[q]["date"] = date
#     print("======================================================")
#     print(bid_data)
#     print(ask_data)
#     print("888888888888888888888888888888888888888888888888888888888888888888888888888888888")
#     c.extend([bid_data])
#     c.extend([ask_data])


#%%
# output_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed'
# output_path_bid = os.path.join(output_path, "LOB_bid.csv")   ## import os
# output_path_ask = os.path.join(output_path, "LOB_ask.csv")   ## import os
# LOB_bid, LOB_ask = process_LOB_data(LOB_list, file_names, output_path_bid, output_path_ask)


#%%
#### ---------------------------------------
# # 设定输入输出路径
#
# output_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed'
# output_path_bid = os.path.join(output_path, "LOB_bid.csv")   ## import os
# output_path_ask = os.path.join(output_path, "LOB_ask.csv")   ## import os
#     # 处理 LOB 数据并输出到 CSV 文件中
# process_LOB_data(LOB_list, file_names, output_path_bid, output_path_ask)

#%%
############################################################################3
"""
output_LOB_data(LOB_list, file_names, order_name,
                    directory_path_bid='E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed/LOB_bid',
                    directory_path_ask='E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed/LOB_ask')

将读取整理的数据 存储到文件夹中。

Input:
:param LOB_list: list of LOB_bid 或者 list of  LOB_ask
:param file_names: list of file_names
:param order_name: type--str; value-- 'bid' or 'ask'
:param directory_path_bid: LOB_bid文件夹的路径
:param directory_path_ask: LOB_ask文件夹的路径

Args:

Return：

"""


def output_LOB_data(LOB_list, order_name, file_names,
                    directory_path_bid='E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed/LOB_bid',
                    directory_path_ask='E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOB_processed/LOB_ask'):

    if order_name == 'bid':
        dire_path = directory_path_bid
    elif order_name == 'ask':
        dire_path = directory_path_ask
    else:
        return print('order_name is wrong!')

    for i, list in enumerate(LOB_list):
        name = file_names[i][10:20] + '.csv'
        df_list = pd.DataFrame(list)
        output_path = os.path.join(dire_path, name)  ## import os # 完整的输出路径
        df_list.to_csv(output_path, index=False)

#%%
# LOB_bid, LOB_ask = process_LOB_data(LOB_list, file_names, output_path_bid, output_path_ask)
#
# output_LOB_data(LOB_ask, file_names,'ask')
# output_LOB_data(LOB_bid, file_names, 'bid')


#%%
def auto_ReadAndOutput(directory, file_names_all):
    i = 0
    j = i + 3
    length = len(file_names_all)
    while j <= length:
        file_names__ = file_names_all[i:j]
        LOB_list, file_names = read_LOBtxt(directory, file_names__)
        LOB_bid, LOB_ask = process_LOB_data(LOB_list, file_names)
        output_LOB_data(LOB_bid, 'bid', file_names)
        output_LOB_data(LOB_ask, 'ask', file_names)

        j += 3
        i += 3
        if i == length:
            break
        elif j > length:
            file_names__ = file_names_all[i:]
            LOB_list, file_names = read_LOBtxt(directory, file_names__)
            LOB_bid, LOB_ask = process_LOB_data(LOB_list, file_names)
            output_LOB_data(LOB_bid, file_names, 'bid')
            output_LOB_data(LOB_ask, file_names, 'ask')
            break
    print("--------------------------over--------------------")

#%%
# directory = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOBs/'
directory = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry_lob/'

file_names_all = os.listdir(directory)  # 获取目录中的所有文件名  # import os

LOB_bid, LOB_ask = auto_ReadAndOutput(directory,file_names_all)



