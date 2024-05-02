


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

list

dataset_bidask6 = split_txtdata_6(list_of_lists, 'average')

df_aveprice = pd.DataFrame(data=dataset_bidask6)
# 导出到CSV文件，这里的index=False表示不导出行索引
df_aveprice.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset6_aveprice.csv', index=False)


import pandas as pd
file_path = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/Tapes/UoB_Set01_2025-01-02tapes.csv'
UoB_Set01 = pd.read_csv(file_path)


# 求均值
# import numpy as np
