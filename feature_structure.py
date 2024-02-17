#%%
#Importing the required libraries
import ast
from datetime import datetime
import pandas as pd

#%%
#Reading the csv file
file_name = "UoB_Set01_2025-01-02LOBs.csv"
df = pd.read_csv(file_name)
date = "2025-01-02"
dt = datetime.strptime(date, "%Y-%m-%d")
df['time'] = df['time']+dt.timestamp()
df['bid_price'] = df['bid_price'].apply(ast.literal_eval)
df['ask_price'] = df['ask_price'].apply(ast.literal_eval)

#%%
#Feature engineering

#价格水平差异（Price level difference）：这是指当前每个价格水平与当前
#平均价格之间的比率差异。这个指标能反映出每个价格水平与市场中间价格的相对位置。
#Price level difference: This metric signifies the relative
# difference between each price level and the current mid-price,
# providing insight into the position of each price level relative
# to the market's median value.
df['bid_level_diff'] = df['max_bid']/df['avg_price']-1
df['ask_level_diff'] = df['min_ask']/df['avg_price']-1


#中间价格变化（Mid price change）：它衡量当前中间价格相对于上一个时间步长的中间价格的变化。
# 这个指标跟踪中间价格随时间的动态变化，这个指标能反映出市场的方向。
#Mid price change: It measures the change in the current mid-price
# compared to the mid-price at the previous time step, tracking
# the mid-price movement over time.
df['mid_price_change'] = df['avg_price'] / df['avg_price'].shift(1) - 1

#深度大小累积和：这个指标表示了每个价格水平上未成交的限价订单数量的累积和，
# 这个指标能反映出不同价格点上的市场流动性。
#Depth size cumulative sum: This represents the cumulative sum of
# the number of outstanding
# limit orders at each price level, offering a measure of market liquidity across different price points.
df['bid_depth_size'] = df['bid_price'].apply(lambda x: x[0][1] if x else None).cumsum()
df['ask_depth_size'] = df['ask_price'].apply(lambda x: x[0][1] if x else None).cumsum()


#订单薄不平衡：这个指标衡量了买卖方订单簿的不平衡程度，这个指标能反映出市场的方向。
#Order book imbalance: This metric measures the degree of imbalance
# between the buy and sell side order books and can be used to predict market direction.
df['order_book_imbalance'] = (df['bid_depth_size'] - df['ask_depth_size']) / (df['bid_depth_size'] + df['ask_depth_size'])

#订单累积深度：这个指标表示了每个价格水平上未成交的限价订单数量的累积和，
#可以用来衡量市场的流动性。
#Order cumulative depth: This metric represents the cumulative sum
# of the number of outstanding limit orders at each price level and can be used to measure market liquidity.
df['bid_cumulative_depth'] = df['bid_price'].apply(lambda x: sum([i[1] for i in x]) if x else None)
df['ask_cumulative_depth'] = df['ask_price'].apply(lambda x: sum([i[1] for i in x]) if x else None)

