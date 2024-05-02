import re
import ast
# 查看字典类数据的key键名 :  字典型数据名.keys()

# 文件路径
file_path = 'D:/python_practice/mini_project/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOBs/UoB_Set01_2025-01-02LOBs.txt'

# 打开并读取文件
with open(file_path, 'r') as file:
    text = file.read()

UoB_Set01 = text

def add_quotes_to_specific_word(s, word):
    pattern = r'\b' + re.escape(word) + r'\b'  # 使用正则表达式为特定单词添加引号，\b 是单词边界，确保整个单词被匹配, 使用 re.escape 来处理 word 中可能包含的任何特殊字符
    return re.sub(pattern, f"'{word}'", s)

data_list_set01 = add_quotes_to_specific_word(UoB_Set01, 'Exch0')


# 分割
data = data_list_set01  # 假设 `data` 是包含您数据的字符串变量
lines = data.split('\n')  # 使用换行符分割字符串
# 使用 ast.literal_eval 将每一行转换为列表
list_of_lists = [ast.literal_eval(line) for line in lines if line]
# list_of_lists[:5]  # 查看

# # -------------------------------------------------------------------------------------------
# ## 假设，数据格式是：分开成两个表————第一列 时间戳， 第二三列 bid价格和数量， 第四五列 ask价格和数量
# def split_txtdata_2(dataset:list):
#     timestamp = []
#     bid_price = []
#     bid_quantity = []
#     ask_price = []
#     ask_quantity = []
#     # 提取时间戳
#     for i in range(len(dataset)):
#         timestamp.append(dataset[i][0])
#
#     # 提取 价格单
#     for i in range(len(dataset)):
#         # 价格单中 买家给出的价格单 a buy list，只记录价格和数量
#         for j in range(len(dataset[i][2][0][1])):
#             bid_price.append(dataset[i][2][0][1][j][0])
#             bid_quantity.append(dataset[i][2][0][1][j][1])
#             # 调用格式 bid_list[第几条字典类型数据]['字典中的key值']
#         for q in range(len(dataset[i][2][1][1])):
#             # 价格单中 卖家给出的价格单 a sell list
#             ask_price.append(dataset[i][2][1][1][q][0])
#             ask_quantity.append(dataset[i][2][1][1][q][1])
#             # 调用格式 ask_list[第几条字典型数据]['字典中的key值']
#
#     dataset = {'timestamp': timestamp, 'bid_price': bid_price, 'bid_quantity': bid_quantity, 'ask_price': ask_price,    'ask_quantity': ask_quantity}
#     print('The type of dataset is', type(dataset))
#     print('The names of the keys are : ', dataset.keys())
#     return dataset
#
# dataset_bidask = split_txtdata_2(list_of_lists)

# 将价格单完全展开，相同时间戳的价格和数量数据 被赋予相同的时间戳【时间戳重复】

def split_txtdata_5(dataset:list):
    timestamp_bid = []
    timestamp_ask = []
    bid_price = []
    bid_quantity = []
    bid = []
    ask_price = []
    ask_quantity = []
    ask = []
    for i in range(len(dataset)):
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
    dataset = {'bid': bid, 'timestamp_bid': timestamp_bid, 'bid_price': bid_price, 'bid_quantity': bid_quantity, 'ask': ask, 'timestamp_ask': timestamp_ask, 'ask_price': ask_price, 'ask_quantity': ask_quantity}
    print('The type of dataset is', type(dataset))
    print('The names of the keys are : ', dataset.keys())
    return dataset

dataset_bidask5 = split_txtdata_5(list_of_lists)

keys_bid = ['timestamp_bid','bid_price','bid_quantity']
bid_dict = {key: dataset_bidask5[key] for key in keys_bid if key in dataset_bidask5}
df_bid = pd.DataFrame(data=bid_dict)

keys_ask = ['timestamp_ask', 'ask_price', 'ask_quantity']
ask_dict = {key: dataset_bidask5[key] for key in keys_ask if key in dataset_bidask5}
df_ask = pd.DataFrame(data=ask_dict)

# 导出到CSV文件，这里的index=False表示不导出行索引
df_bid.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset5_bid.csv', index=False)
df_ask.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset5_ask.csv', index=False)



# -----------------------------------------------------------------------
# 一共有三列：第一列 时间戳， 第二列 bid的价格(均值/最大值/最小值)，第三列 ask的价格(均值/最大值/最小值)

def split_txtdata_6(dataset:list, price_type:str):
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


dataset_bidask6 = split_txtdata_6(list_of_lists, 'average')

df_aveprice = pd.DataFrame(data=dataset_bidask6)
# 导出到CSV文件，这里的index=False表示不导出行索引
df_aveprice.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset6_aveprice.csv', index=False)
