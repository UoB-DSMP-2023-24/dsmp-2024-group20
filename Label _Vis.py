import pandas as pd
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(14, 7))
df = pd.read_csv('process_data_weight/UoB_Set01_2025-01-02LOBs.csv')
df = df.dropna()
df =df.iloc[:500]

# 绘制价格线
ax.plot(df['time_window'], df['avg_price'], label='Price')

# 根据标签着色背景
for index, row in df.iterrows():
    if row['label'] == 1:
        ax.axvspan(row['time_window'], row['time_window']+1, color='green', alpha=0.3)
    elif row['label'] == 2:
        ax.axvspan(row['time_window'], row['time_window']+1, color='red', alpha=0.3)
    # 对于标签为0的数据点，不进行背景着色

# 设置标题和坐标轴标签
ax.set_xlabel('time')
ax.set_ylabel('price')

# 添加图例
ax.legend()
plt.savefig('price_trend.png')
# 显示图表
plt.show()