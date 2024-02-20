import os

# 正确的Windows路径字符串格式
base_dir = r'D:\python_practice\mini_project\mini_projectB_sample_0129_2024\Problem B data\JPMorgan_Set01\LOBs'

# 初始化一个列表来存储文件内容
data_dir = []

# 遍历文件夹中的每个文件
for file in os.listdir(base_dir):
    if file.endswith('.txt'):  # 确保只处理.txt文件
        file_path = os.path.join(base_dir, file)  # 获取完整的文件路径
        with open(file_path, 'r') as f:  # 打开文件
            data = f.read()  # 读取文件内容
            data_dir.append(data)  # 将内容添加到列表中

