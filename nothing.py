

## 年月日
def insert_date(list, position, date):  ## 时间--年月日
    # date = pd.to_datetime(file_name[10:20])
    # date = file_name[10:20]
    list.insert(position, date)
    return list

## 生成文件路径  ## 可能需要调整一下
import os
file_name = []
if file_name.endswith('.txt'):
    file_path = os.path.join(directory, file_name)

## 第33行和54行的地方 给 bidsdf 和 asksdf 增加一列时间？
def timestamp_to_time(data, filenames, start_time=[8, 0, 0]):
    """
    Converts a timestamp to a time
    :param data: pandas dataframe with timestamps 【！】 timestamp column's name : "time_window"
    :param filenames: come from the read_csvfile_all3()
    :param start_time: list; default value [8, 0, 0]
    :return: pandas dataframe with time: YYYY-MM-DD HH:mm:ss.ms
    """
    for filename in filenames:
        data['time'] = pd.to_datetime(data['time_window'], unit='s')
        data['time'] = data['time'].apply(lambda x: x.replace(year=int(filename[10:14]),
                                                              month=int(filename[15:17]),
                                                              day=int(filename[18:20])))
        data['time'] += Timedelta(hours=start_time[0], minutes=start_time[1], seconds=start_time[2])

    return data



## 使用下面的代码，合并df数据：
import pandas as pd
df1, df2, df3 = [],[],[]
df = pd.concat([df1, df2, df3]).reset_index(drop=True)


## 重新命名
original_file_name.split('_')  # 文件名格式: "UoB_Set01_2025-01-02LOBs.csv"


