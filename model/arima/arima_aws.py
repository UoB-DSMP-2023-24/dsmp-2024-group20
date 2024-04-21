import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

stock_file = '/Users/fanxinwei/Desktop/code/git_repo/dsmp-2024-group20/input_data/total_lob_30.csv'
df = pd.read_csv(stock_file)
# 解析日期
df['date'] = pd.to_datetime(df['date'])
# 将时间窗口转换为timedelta（时间窗口以秒为单位），并设置每天的起始时间为8:00
df['datetime'] = df['date'] + pd.to_timedelta('8 hours') + pd.to_timedelta(df['time_window'], unit='s')
# 设置新的日期时间为索引
df.set_index('datetime', inplace=True)

price = df['avg_price']

train_data = price[:-10]
test_data = price[-10:]
train_data_diff = train_data.diff(1)
train_data_diff = train_data_diff.dropna()


## 自动寻找最佳ARIMA模型
### remember to check y is base on with training data: train_avg ? or train_wavg?
auto_model = pm.auto_arima( train_data, start_p=0, start_q=0,
                            test='adf',
                            max_p=10, max_q=10,
                            seasonal=False,   ## if seasonal=True, need to add "start_P = " and "D="
                            # although "False" is been choosen, must set the "m="( but "m=" does't work)
                            suppress_warnings= 'True',
                            information_criterion= 'aic', ## using aic to choose the best model
                            error_action='ignore',
                            stepwise= False,    ## “True”(default) may cannot find the best way, but "False" will waste more time.
                            trace= True   ## print the detail of the searching process
                            )
auto_model.fit

print(auto_model.summary())