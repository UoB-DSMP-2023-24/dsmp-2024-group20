# 预测的股票价格序列
predicted_prices = [297.486500, 298.937643, 299.598436, 300.119995, 300.457963]

# 初始资本和每次交易的股票数量
initial_capital = 1000
stock_quantity = 1

# 模拟交易
capital = initial_capital
stock_owned = 0
transactions = []

for i in range(len(predicted_prices) - 1):
    today_price = predicted_prices[i]
    tomorrow_price = predicted_prices[i + 1]

    # 如果明天的价格预测高于今天，尝试买入一股
    if tomorrow_price > today_price and capital >= today_price:
        capital -= today_price
        stock_owned += stock_quantity
        transactions.append(f"Day {i + 1}: Bought 1 stock at {today_price}")
    # 如果明天的价格预测低于今天，尝试卖出一股
    elif tomorrow_price < today_price and stock_owned >= stock_quantity:
        capital += today_price
        stock_owned -= stock_quantity
        transactions.append(f"Day {i + 1}: Sold 1 stock at {today_price}")

# 最后一天，如果持有股票，则全部卖出
last_day_price = predicted_prices[-1]
if stock_owned > 0:
    capital += stock_owned * last_day_price
    stock_owned = 0
    transactions.append(f"Day {len(predicted_prices)}: Sold {stock_quantity} stock at {last_day_price}")

# 计算最终资本和利润
final_capital = capital
profit = final_capital - initial_capital

transactions, final_capital, profit
