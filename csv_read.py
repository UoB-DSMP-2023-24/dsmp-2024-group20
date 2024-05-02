# 读入文件夹中的所有的 csv格式的数据文件

import pandas as pd
import os  # 在自定义函数中  read_csvfile_all

# 设置目录路径
directory = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/Tapes/'


def read_csvfile_all(directory):
    # import os
    data = []
    file_name_ = os.listdir(directory)  # 获取目录中的所有文件名
    for i in range(len(file_name_)):
        file_path = directory + file_name_[i]
        data.append({file_name_[i]: pd.read_csv(file_path)})
    # # 一个数据框的形式
    # dataframe = pd.concat(data)
    return data


data_csv = read_csvfile_all(directory)


# 测试
# directory_trytry = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry/'
# # file_names_trytry = os.listdir(directory_trytry)
# csvfiles = read_csvfile_all(directory_trytry)

# --------------------------------------------------------------------------
# 一个数据框的形式
def read_csvfile_all2(directory):
    # import os
    data = []
    file_name_ = os.listdir(directory)  # 获取目录中的所有文件名
    for i in range(len(file_name_)):
        file_path = directory + file_name_[i]
        data.append(pd.read_csv(file_path))
    print('the type of data is', type(data))
    # 一个数据框的形式
    dataframe = pd.concat(data)
    return dataframe

# # 测试
# directory_trytry = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry/'
# # file_names_trytry = os.listdir(directory_trytry)
# csvfiles = read_csvfile_all2(directory_trytry)


import pandas as pd
import os
#################################################################################################
# 所有文件读取到一个DataFrame， 里面有三列数据： 第一列 timestamp， 第二列 price， 第三列 quantity
def read_csvfile_all3(directory):
    dataframes = []  # 用于存储每个 CSV 文件的 DataFrame
    file_names = os.listdir(directory)  # 获取目录中的所有文件名
    column_names = ['timestamp', 'price', 'quantity']
    for file_name in file_names:
        if file_name.endswith('.csv'):  # 先检查是否是 CSV 文件
            file_path = os.path.join(directory, file_name)  # 正确地构建文件路径
            data = pd.read_csv(file_path, header=None, names=column_names, usecols=[0, 1, 2])
            dataframes.append(data)  # 将 DataFrame 添加到列表中

    # 在所有 CSV 文件处理完成后，合并所有 DataFrame
    if dataframes:  # 确保列表不为空
        dataframe = pd.concat(dataframes, ignore_index=True)
        print('The type of data is', type(dataframe))
        return dataframe
    else:
        print('No CSV files found or DataFrame is empty.')
        return pd.DataFrame()  # 如果没有找到 CSV 文件或列表为空，则返回空的 DataFrame

directory_trytry = 'E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry/'
data_csv = read_csvfile_all3(directory_trytry)
