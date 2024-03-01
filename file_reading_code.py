# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Vishal Chavan
"""

import pandas as pd
#%%
lob = []
with open("C:/Users/Vishal Chavan/Desktop/Uni/subjects/group project/UoB_Set01_2025-01-02LOBs.txt", 'r') as file:
    for i, line in enumerate(file):
        lob.append(list(line.split('[[')))
        if i == 30000:
            break
#%%
import ast
bids = []
time = []
n = []
m_price=[]
number_of_prices = 5
for line in lob:
    try:
        if len(line)>2:
            dt = ast.literal_eval('[['+list(line[2].split("]]], ['ask', "))[0]+']]')
            bids.append(dt)
            time.append(line[0].split(',')[0][1:])
            n.append(len(dt))
            m_price.append(bids[len(bids)-1][0][0])
    except Exception as e:
        print(f"Error processing line: {line}. Error: {e}")
bidsdf = pd.DataFrame({'time':time,
                       "bid_price":[i[0:number_of_prices] for i in bids],
                       "n_bid_prices":n,
                       "max_bid":m_price})
#%%
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
asksdf = pd.DataFrame({'time':time,
                       "ask_price":[i[0:number_of_prices] for i in asks],
                       "n_ask_price":n,
                       "min_ask":m_price})
#%%
df = pd.merge(bidsdf,asksdf,on='time',how='outer').dropna()
df['market_price'] = (df['max_bid']+df['min_ask'])/2
#%%
tapes = pd.read_csv("C:/Users/Vishal Chavan/Desktop/Uni/subjects/group project/UoB_Set01_2025-01-02tapes.csv",
                    nrows=999)
#%%
df[['max_bid','min_ask']].to_csv("maxmin.csv")
#%%