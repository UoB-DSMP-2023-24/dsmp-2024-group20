# 调整txt类型的数据文件的数据格式，并将文件读入———— 以list的格式


import re
def add_quotes_to_specific_word(s, word):
    # 使用正则表达式为特定单词添加引号
    # \b 是单词边界，确保整个单词被匹配
    # 使用 re.escape 来处理 word 中可能包含的任何特殊字符
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern, f"'{word}'", s)

# # 测试字符串
# test_str = "[0.000, Exch0, [['bid', []], ['ask', []]]]"
# # 转换字符串，只为"Exch0"添加引号
# corrected_str = add_quotes_to_specific_word(test_str, 'Exch0')
# print(corrected_str)

# 文件路径
file_path = 'D:/python_practice/mini_project/mini_projectB_sample_0129_2024/Problem B data/JPMorgan_Set01/LOBs/UoB_Set01_2025-01-02LOBs.txt'

# 打开并读取文件
with open(file_path, 'r') as file:
    text = file.read()

data = add_quotes_to_specific_word(text, 'Exch0')
type(data)

# 分割
#  `data` 是包含数据的字符串变量
# 使用换行符分割字符串
lines = data.split('\n')
import ast
# 使用 ast.literal_eval 将每一行转换为列表
list_of_lists = [ast.literal_eval(line) for line in lines if line]
        # # 遍历 lines 中的每一行
        # for line in lines:
        #     if line:  # 检查 line 是否非空
        #         # 使用 ast.literal_eval 将字符串转换为相应的 Python 对象
        #         # 并将转换后的对象添加到 list_of_lists 中
        #         list_of_lists.append(ast.literal_eval(line))

list_of_lists[:5]

# --------------------------------------------------------------------
# 集成上述过程为函数







