# 将txt格式的文件，读取成list格式，转换成DataFrame格式

import re
import ast
import pandas as pd
# 查看字典类数据的key键名 :  字典型数据名.keys()

# 文件路径
file_path = 'D:/python_practice/mini_project/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOBs/UoB_Set01_2025-01-02LOBs.txt'

# 打开并读取文件
with open(file_path, 'r') as file:
    text = file.read()

# UoB_Set01 = text

def add_quotes_to_specific_word(s, word):
    pattern = r'\b' + re.escape(word) + r'\b'  # 使用正则表达式为特定单词添加引号，\b 是单词边界，确保整个单词被匹配, 使用 re.escape 来处理 word 中可能包含的任何特殊字符
    return re.sub(pattern, f"'{word}'", s)

data_list = add_quotes_to_specific_word(text, 'Exch0')


# 分割
lines = data_list.split('\n')  # 使用换行符分割字符串
# 使用 ast.literal_eval 将每一行转换为列表
list_of_lists = [ast.literal_eval(line) for line in lines if line]
# list_of_lists[:5]  # 查看







# -----------------------------------------------------------------------
# 转换成 数据框格式
# 一共有三列：第一列 时间戳， 第二列 bid的价格(均值/最大值/最小值)，第三列 ask的价格(均值/最大值/最小值)
# 使用的是原本的“split_txtdata_6”函数

def split_txtdata(dataset:list, price_type:str):
    if not price_type or price_type not in ['average', 'max', 'min']:
        print(f"Error: {price_type} is not a valid price type.")
        print('Please enter str_data : average or max or min. And remember to enter ''.')
        return
    timestamp = []
    bid = []
    price_bid = 0.0
    ask = []
    price_ask = 0.0
    # 提取时间戳
    for i in range(len(dataset)):
        timestamp.append(dataset[i][0])
        if price_type == 'average':
            if len(dataset[i][2][0][1]) == 0:
                bid.append(0)
            else:
                for j in range(len(dataset[i][2][0][1])):
                    price_bid += (dataset[i][2][0][1][j][0])
                price_bid = price_bid / len(dataset[i][2][0][1])
                bid.append(price_bid)
        elif price_type == 'max':
            for j in range(len(dataset[i][2][0][1])):
                price_bid = max(price_bid, dataset[i][2][0][1][j][0])
            bid.append(price_bid)
        else:
            for j in range(len(dataset[i][2][0][1])):
                price_bid = min(price_bid, dataset[i][2][0][1][j][0])
            bid.append(price_bid)
# --------------------------------
        if price_type == 'average':
            if len(dataset[i][2][1][1]) == 0:
                ask.append(0)
            else:
                for q in range(len(dataset[i][2][1][1])):
                    price_ask += (dataset[i][2][1][1][q][0])
                price_ask = price_ask / len(dataset[i][2][1][1])
                ask.append(price_ask)
        elif price_type == 'max':
            for q in range(len(dataset[i][2][1][1])):
                price_ask = max(price_ask, dataset[i][2][1][1][q][0])
            ask.append(price_ask)
        else:
            for q in range(len(dataset[i][2][1][1])):
                price_ask = min(price_ask, dataset[i][2][1][1][q][0])
            ask.append(price_ask)
        price_ask = 0.0
        price_bid = 0.0
    dataset = {'timestamp': timestamp, 'bid': bid, 'ask': ask}
    print('The type of dataset is', type(dataset))
    print('The names of the keys are : ', dataset.keys())
    return dataset


# 测试函数性能
dataset = split_txtdata(list_of_lists, 'average')
df_ = pd.DataFrame(data=dataset)


# # 导出到CSV文件，这里的index=False表示不导出行索引
# df_.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset6_aveprice.csv', index=False)


