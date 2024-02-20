# 没成--图像太乱
# import matplotlib.pyplot as plt
# # 数据点
#
# # dict_keys(['timestamp', 'bid_price', 'bid_quantity', 'ask_price', 'ask_quantity'])
# # x = dataset_bidask['timestamp']
# x = dataset_bidask['bid_quantity']
# y = dataset_bidask['bid_price']
# color = dataset_bidask['bid_quantity']
#
# fig, ax = plt.subplots()
# # 根据 bid_quantity 的数值改变散点的颜色，你可以使用 c 参数而不是 color，并且指定一个颜色映射（colormap）
# # c 参数将基于 bid_quantity 的数值为每个点指定一个颜色，颜色将从 viridis 颜色映射中选取
# # 注意： x, y, color 的数据长度必须匹配(一样)
# ax.scatter(x, y, label = 'bid', c = color, cmap = 'viridis') # We generate a scatterplot of the data on the axes.-- 在（坐标）轴上生成数据的散点图
#
# plt.xlabel('time') # 给坐标轴命名-- 横轴--“x”
# plt.ylabel('price') # 给坐标轴命名-- 横轴--“y”
#
# # # 添加标题
# # plt.title('Simple Plot')
# # 显示图形
# plt.show()
# # 清除画布(空图)  plt.clf()  ; 清除画布(空图)上的图像 cla()  plt.cla()


# ______________________________________________________________________________________________
# 时间戳的数量和 bid价格单的数量不一致，排除这个方法
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # 假设您的原始数据如下
# timestamps = [0.0, 0.279, 1.333, 1.581, 1.643, 1.736, 1.984, 2.015, 2.139, 2.697, 3.069, 3.131, 3.255, 3.286, 3.441, 3.782, 3.813, 3.906, 4.216, 4.34]
# prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195]
# quantities = [200, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
#
# # 将数据转换为DataFrame
# df = pd.DataFrame({
#     'Time': timestamps,
#     'Price': prices,
#     'Quantity': quantities
# })
#
# # 设置图表和轴
# fig, ax1 = plt.subplots()
#
# # 绘制价格随时间变化的曲线
# ax1.plot(df['Time'], df['Price'], 'b-')
# ax1.set_xlabel('Time (hours)')
# ax1.set_ylabel('Price', color='b')
# ax1.tick_params('y', colors='b')
#
# # 创建一个共享x轴的新轴对象用于数量
# ax2 = ax1.twinx()
# ax2.scatter(df['Time'], df['Quantity'], color='r')
# ax2.set_ylabel('Quantity', color='r')
# ax2.tick_params('y', colors='r')
#
# plt.title('Price and Quantity over Time')
# plt.show()


# ------------------------------------------------------------------
# 深度图
import matplotlib.pyplot as plt

# 假设你有两个列表或numpy数组，一个是价格，一个是相应的数量
# prices = dataset_bidask['bid_price', 'ask_price']  # 价格数据
# quantities = dataset_bidask['bid_quantity', 'ask_quantity']  # 对应的数量数据

# 通常买单和卖单会分开处理
bids_prices = dataset_bidask['bid_price']  # 买单价格
bids_quantities = dataset_bidask['bid_quantity']  # 买单数量
asks_prices = dataset_bidask['ask_price']  # 卖单价格
asks_quantities = dataset_bidask['ask_quantity']  # 卖单数量

# 绘制深度图
plt.fill_between(bids_prices, bids_quantities, step="post", color="green", alpha=0.6)
plt.fill_between(asks_prices, asks_quantities, step="post", color="red", alpha=0.6)

plt.title("LOB Depth Chart")
plt.xlabel("Price")
plt.ylabel("Cumulative Quantity")
plt.show()

