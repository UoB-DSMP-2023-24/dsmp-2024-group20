import pandas as pd
import datetime
timestamp = 30588.568
timestamp2 = 11334.034
seconds = int(timestamp)
milliseconds = int(timestamp - seconds)

dt = datetime.datetime.fromtimestamp(timestamp)
dt2 = datetime.datetime.fromtimestamp(timestamp2)


print('日期时间:', dt)
print('日期时间:', dt2)



# 时间戳转换成时间
def timestamp_to_time(timestamp):
    import datetime
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt

# 根据 csv_read.py文件的 read_csvfile_all3(directory)函数，读取的datacsv02 数据框
def read_csvfile_all3(directory):
    import os

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




column_names = ['timestamp', 'price', 'quantity']
df = pd.read_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/trytry/UoB_Set01_2025-01-03tapes.csv', header=None, names=column_names, usecols=[0, 1, 2])
# 应用函数到每个元素
time = df['timestamp'].apply(timestamp_to_time)
time2 = df['timestamp2'].apply(timestamp_to_time)


# 导出文件  csv格式
# df.to_csv('E:/Bristol_tb2/mini_projectB/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/dataset0001.csv', index=False)

import matplotlib.pyplot as plt

df_2 = pd.DataFrame({'timestamp': time, 'price':df['price'], 'quantity':df['quantity']})
# 绘制分时图
plt.plot(df['timestamp'], df['price'])
plt.title('Price vs. Time Chart')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)  # 显示网格
plt.show()  # 显示图表