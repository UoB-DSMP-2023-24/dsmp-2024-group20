import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

result_map = {}

for i in range(2, 6):
    file_dir = '/Users/fanxinwei/Desktop/code/git_repo/dsmp-2024-group20/'
    stock_file = file_dir + 'feature_extract/time_window_outputs/2025-01-02LOBs_' + str(i) + '_seconds.csv'
    df = pd.read_csv(stock_file)
    start_date = pd.to_datetime('2025-01-02 08:00:00')
    df['actual_datetime'] = start_date + pd.to_timedelta(df['time_window'], unit='s')
    df.set_index('actual_datetime', inplace=True)
    price = df['avg_price']

    steps = 100

    train_data = price[:-steps]
    test_data = price[-steps:]
    train_data_diff = train_data.diff(1)
    train_data_diff = train_data_diff.dropna()

    model = ARIMA(train_data, order=(4, 0, 1))
    results = model.fit()
    forecast = results.forecast(steps=steps)

    forecast_df = pd.DataFrame(forecast.values, index=test_data.index)

    differences = forecast_df[0] - test_data
    total_difference = differences.abs().sum()
    result_map[i] = total_difference

print(result_map)