# lob_1 = list_of_lists[:5]
#
# '''
# 每一条数据的 第一个是 时间戳；第二个是 股票名字/机构名字；第三个是 价格单。
# 对于第三个列表——价格单：
#     第一个是 买家给出的价格单；第二个是 卖家给出的价格单
#     使用len()来判断两个价格单分别是多长。
#     长度减一等于 每个价格对应的数量。
# '''
# timestamp = []
# name = []
# # price_quantity = []
#
# # 提取时间戳
# for i in range(len(lob_1)):
#     timestamp.append(lob_1[i][0])
#     name.append(lob_1[i][1])
#     # price_quantity.append(lob_1[i][2])
#
# # 提取 价格单
# bid_list = []
# ask_list = []
# for i in range(len(lob_1)):
#     # 价格单中 买家给出的价格单 a buy list
#     bid_list.append( {'timestamp': timestamp[i], 'PriceAndQuantity': lob_1[i][2][0][1]} )
#         # 调用格式 bid_list[第几条字典类型数据]['字典中的key值']
#     # 价格单中 卖家给出的价格单 a sell list
#     ask_list.append( {'timestamp': timestamp[i], 'PriceAndQuantity':lob_1[i][2][1][1]})
#         # 调用格式 ask_list[第几条字典型数据]['字典中的key值']

####################################################################################################


## 返回值： [列表 [bid列表]， [ask列表] ]
## bid列表：[列表 {字典'PriceAndQuantity':[ [价格1,数量1], [价格2, 数量2] ], 'timestamp': 数字 }]
def split_txtdata( dataset:list):
    timestamp = []
    name = []
    bid_list = []
    ask_list = []

    # 提取时间戳
    for i in range(len(dataset)):
        timestamp.append(dataset[i][0])
        name.append(dataset[i][1])

    # 提取 价格单
    for i in range(len(dataset)):
        # 价格单中 买家给出的价格单 a buy list
        bid_list.append({'timestamp': timestamp[i], 'PriceAndQuantity': dataset[i][2][0][1]})
        # 调用格式 bid_list[第几条字典类型数据]['字典中的key值']
        # 价格单中 卖家给出的价格单 a sell list
        ask_list.append({'timestamp': timestamp[i], 'PriceAndQuantity': dataset[i][2][1][1]})
        # 调用格式 ask_list[第几条字典型数据]['字典中的key值']

    dataset =[bid_list, ask_list]
    return dataset

dataset_justtry = list_of_lists[:20]
dataset_bidask = split_txtdata(dataset_justtry)
## 返回值格式： [列表 [bid列表]， [ask列表] ]
## bid列表：[列表 {字典'PriceAndQuantity':[ [价格1,数量1], [价格2, 数量2] ], 'timestamp': 数字 }]


# ————————————————————————————————————————————————————————————————————————————————————————————————
## 假设，数据格式是：分开成两个表————第一列 时间戳， 第二三列 bid价格和数量， 第四五列 ask价格和数量
def split_txtdata_2(dataset:list):
    timestamp = []
    bid_price = []
    bid_quantity = []
    ask_price = []
    ask_quantity = []


    # 提取时间戳
    for i in range(len(dataset)):
        timestamp.append(dataset[i][0])

    # 提取 价格单
    for i in range(len(dataset)):
        # 价格单中 买家给出的价格单 a buy list，只记录价格和数量
        for j in range(len(dataset[i][2][0][1])):
            bid_price.append(dataset[i][2][0][1][j][0])
            bid_quantity.append(dataset[i][2][0][1][j][1])
            # 调用格式 bid_list[第几条字典类型数据]['字典中的key值']
        for q in range(len(dataset[i][2][1][1])):
            # 价格单中 卖家给出的价格单 a sell list
            ask_price.append(dataset[i][2][1][1][q][0])
            ask_quantity.append(dataset[i][2][1][1][q][1])
            # 调用格式 ask_list[第几条字典型数据]['字典中的key值']

    dataset = {'timestamp': timestamp, 'bid_price': bid_price, 'bid_quantity': bid_quantity, 'ask_price': ask_price,    'ask_quantity': ask_quantity}
    return dataset

dataset_bidask2 = split_txtdata_2(dataset_justtry)

########################################################################################
## 假设，数据格式是：分开成两个表————第一列 时间戳， 第二列 价格， 第三列 数量

def split_txtdata_3(dataset:list):
    timestamp = []
    bid_list = []
    bid_price = []
    bid_quantity = []
    ask_list = []
    ask_price = []
    ask_quantity = []


    # 提取时间戳
    for i in range(len(dataset)):
        timestamp.append(dataset[i][0])

    # 提取 价格单
    for i in range(len(dataset)):
        # 价格单中 买家给出的价格单 a buy list，只记录价格和数量
        for j in range(len(dataset[i][2][0][1])):
            bid_price.append(dataset[i][2][0][1][j][0])
            bid_quantity.append(dataset[i][2][0][1][j][1])
            # 调用格式 bid_list[第几条字典类型数据]['字典中的key值']
        for q in range(len(dataset[i][2][1][1])):
            # 价格单中 卖家给出的价格单 a sell list
            ask_price.append(dataset[i][2][1][1][q][0])
            ask_quantity.append(dataset[i][2][1][1][q][1])
            # 调用格式 ask_list[第几条字典型数据]['字典中的key值']

    dataset = {'timestamp': timestamp, 'bid_price': bid_price, 'bid_quantity': bid_quantity, 'ask_price': ask_price,    'ask_quantity': ask_quantity}
    bid_list = [bid_price, bid_quantity]
    ask_list = [ask_price, ask_quantity]
    # bid_list 和 ask_list 的第一个列表都是价格，第二个列表都是数量。
    return bid_list, ask_list

dataset_bid3, dataset_ask3 = split_txtdata_3(dataset_justtry)


##############################################################
# 第一列 时间戳，第二列 bid订单[ [价格, 数量],[价格, 数量],...,[价格，数量] ]，第三列 ask定价单[[价格, 数量],...,[价格, 数量]]

def split_txtdata_4(dataset:list):
    timestamp = []
    bid = []
    price_bid = ()
    ask = []
    price_ask = ()
    # 提取时间戳
    for i in range(len(dataset)):
        timestamp.append(dataset[i][0])

    # 提取 价格单
    for i in range(len(dataset)):
        # 价格单中 买家给出的价格单 a buy list，只记录价格和数量
        bid.extend([dataset[i][2][0][1]])
        # 价格单中 卖家给出的价格单 a sell list
        ask.extend([dataset[i][2][1][1]])

    dataset = {'timestamp': timestamp, 'bid': bid, 'ask': ask}
    print('The type of dataset is', type(dataset))
    print('The names of the keys are : ', dataset.keys())
    return dataset

dataset_bidask4 = split_txtdata_4(dataset_justtry)


import pandas as pd

# 创建DataFrame
df = pd.DataFrame(dataset_bidask4)

# 导出到CSV文件，这里的index=False表示不导出行索引
df.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset_bidask4.csv', index=False)

# -----------------------------------------------------------------------------
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

dataset_bidask5 = split_txtdata_5(dataset_justtry)

keys_bid = ['timestamp_bid','bid_price','bid_quantity']
bid_dict = {key: dataset_bidask5[key] for key in keys_bid if key in dataset_bidask5}
df_bid = pd.DataFrame(data=bid_dict)

keys_ask = ['timestamp_ask', 'ask_price', 'ask_quantity']
ask_dict = {key: dataset_bidask5[key] for key in keys_ask if key in dataset_bidask5}
df_ask = pd.DataFrame(data=ask_dict)

# 导出到CSV文件，这里的index=False表示不导出行索引
df_bid.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset5_bid.csv', index=False)
df_ask.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset5_ask.csv', index=False)

# --------------------------------------------------------------------------------
##############################################################
# 第一列 时间戳，第二列 bid订单对应时间戳的平均价格，第三列 ask对应时间戳的平均价格

# price_type: average / max / min

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
                for j in range(len(dataset[i][2][1][1])):
                    price_ask += (dataset[i][2][1][1][j][0])
                price_ask = price_ask / len(dataset[i][2][1][1])
                ask.append(price_ask)
        elif price_type == 'max':
            for j in range(len(dataset[i][2][1][1])):
                price_ask = max(price_ask, dataset[i][2][1][1][j][0])
            ask.append(price_ask)
        else:
            for j in range(len(dataset[i][2][1][1])):
                price_ask = min(price_ask, dataset[i][2][1][1][j][0])
            ask.append(price_ask)
        price_ask = 0.0
        price_bid = 0.0

    dataset = {'timestamp': timestamp, 'bid': bid, 'ask': ask}
    print('The type of dataset is', type(dataset))
    print('The names of the keys are : ', dataset.keys())
    return dataset

dataset_bidask6 = split_txtdata_6(dataset_justtry,'average')
