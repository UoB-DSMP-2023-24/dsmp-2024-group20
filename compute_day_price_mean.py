import os
import pandas as pd
import re

# 指定包含CSV文件的文件夹路径
folder_path = '/Users/fanxinwei/Desktop/code/tapes'
# 创建一个空的DataFrame来存储所有天的平均值
data = {'date': [], 'avg_value': []}
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是CSV文件
    if filename.endswith('.csv'):
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 计算第二列的平均值
        avg_value = df.iloc[:, 1].mean()

        # 使用正则表达式提取日期
        date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
        match = date_pattern.search(filename)
        if match:
            date = match.group(1)
            data['date'].append(date)
            data['avg_value'].append(avg_value)
        else:
            continue  # 如果文件名中没有日期，则跳过该文件

# 创建DataFrame
result_df = pd.DataFrame(data)

# 按日期正序排列
result_df.sort_values(by='date', inplace=True)

try:
    result_df.to_csv('/Users/fanxinwei/Desktop/code/git_repo/dsmp-2024-group20/ARIMA/avg_data.csv', index=False)
    print("数据保存成功！")
except Exception as e:
    print("数据保存失败:", e)