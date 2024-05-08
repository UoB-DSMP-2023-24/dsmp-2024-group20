# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:01:38 2024
@author: Vishal Chavan
"""
import pandas as pd
import json
config = json.load(open("config.json"))
lob = []
import ast
bids = []
time = []
n = []
m_price=[]
number_of_prices = 10
bidsdf = pd.DataFrame({'time':time,
                       "bid_price":[i[0:number_of_prices] for i in bids],
                       "n_bid_prices":n,
                       "max_bid":m_price})
asks = []
time = []
n = []
m_price = []
for line in lob:
    try:
        if len(line)>3:
            dt = ast.literal_eval('[['+list(line[3].split("]]]]]"))[0]+']]')
            asks.append(dt)
            time.append(line[0].split(',')[0][1:])
            n.append(len(dt))
            m_price.append(asks[len(asks)-1][0][0])
    except Exception as e:
        print(f"Error processing line: {line}. Error: {e}")
        bids.append([[None]])
        time.append(line[0].split(',')[0][1:])
        n.append(0)
        m_price.append(None)
asksdf = pd.DataFrame({'time':time,
                       "ask_price":[i[0:number_of_prices] for i in asks],
                       "n_ask_price":n,
                       "min_ask":m_price})
df = pd.merge(bidsdf,asksdf,on='time',how='outer').dropna()
df['market_price'] = (df['max_bid']+df['min_ask'])/2
import numpy as np
df['Duration'] = df['time'].astype(float).diff(1)
df['maxbid_diff'] = df['max_bid'].astype(int).diff()
df['minask_diff'] = df['min_ask'].astype(int).diff()
df['marketprice_diff'] = df['market_price'].astype(float).diff()
df['bid_cumulative_depth'] = df['bid_price'].apply(lambda x: sum([i[1] for i in x]) if x else None)
df['ask_cumulative_depth'] = df['ask_price'].apply(lambda x: sum([i[1] for i in x]) if x else None)
df['bid_ask_depth_diff'] = df['bid_cumulative_depth'] - df['ask_cumulative_depth']
min_min_ask =  np.min(df['min_ask'])
min_market_price = np.min(df['market_price'])
min_bid_ask_depth_diff = np.min(df['bid_ask_depth_diff'])
min_marketprice_diff = np.min(df['marketprice_diff'])
max_max_bid =  np.max(df['max_bid'])
max_market_price = np.max(df['market_price'])
max_bid_ask_depth_diff = np.max(df['bid_ask_depth_diff'])
max_marketprice_diff = np.max(df['marketprice_diff'])
mean_min_ask =  np.mean(df['min_ask'])
mean_max_bid =  np.mean(df['max_bid'])
mean_market_price = np.mean(df['market_price'])
mean_bid_ask_depth_diff = np.mean(df['bid_ask_depth_diff'])
mean_marketprice_diff = np.mean(df['marketprice_diff'])
median_min_ask =  np.median(sorted(df['min_ask']))
median_max_bid =  np.median(sorted(df['max_bid']))
median_market_price = np.median(sorted(df['market_price']))
median_bid_ask_depth_diff = np.median(sorted(df['bid_ask_depth_diff']))
median_marketprice_diff = np.median(sorted(df['marketprice_diff']))
from scipy import stats
mode_min_ask =  stats.mode(sorted(df['min_ask']))
mode_max_bid =  stats.mode(sorted(df['max_bid']))
mode_market_price = stats.mode(sorted(df['market_price']))
mode_bid_ask_depth_diff = stats.mode(sorted(df['bid_ask_depth_diff']))
mode_marketprice_diff = stats.mode(sorted(df['marketprice_diff']))
df_market_prices = df['market_price'].drop_duplicates(keep='first')
df_durations = round(df['Duration'],3).drop_duplicates(keep='first')