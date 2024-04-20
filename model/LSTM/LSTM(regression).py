#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

#%%
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).to(x.device)

        # Initialize cell state
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_dim)).to(x.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.linear(out[:, -1, :])
        return out
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).numpy()

#%%

df = pd.read_csv('../../output_data/UoB_Set01_2025-01-02LOBs.csv')
df = df.dropna()
# df = df.iloc[:8000]

start_date = pd.to_datetime('2025-01-02 08:00:00')
df['actual_datetime'] = start_date + pd.to_timedelta(df['time_window'], unit='s')
df.set_index('actual_datetime', inplace=True)
feature=df[['max_bid', 'min_ask', 'avg_price','avg_price_change',
            'bid_level_diff', 'ask_level_diff', 'bid_cumulative_depth', 'ask_cumulative_depth','bid_ask_depth_diff']]
target=df['l_t']

scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature.values)  # 对特征进行标准化

# 数据划分
train_size = int(len(feature) * 0.8)
X_train, X_test = feature.iloc[:train_size], feature.iloc[train_size:]
y_train, y_test = target.iloc[:train_size], target.iloc[train_size:]

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 注意：我们使用训练集的参数来转换测试集

# 转换为适合LSTM的数据格式

time_steps = 10  # 可以根据需要调整
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, time_steps)

# 转换为torch张量
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float).view(-1, 1)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float).view(-1, 1)

# DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=False)
#%%
# 初始化模型
model = LSTMModel(input_dim=feature.shape[1], hidden_dim=64, num_layers=2, output_dim=1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# 预测
predictions = predict(model, X_test_tensor)
#%%

#%%
import torch
import torch.nn.functional as F

# compute mse loss for the predictions
predictions_tensor = torch.from_numpy(predictions).float()
mse = F.mse_loss(predictions_tensor, y_test_tensor)
print("MSE:", mse.item())
# compute rmse
rmse = torch.sqrt(mse)
print("RMSE:", rmse.item())
# compute r2 score
ssr = torch.sum((predictions_tensor - y_test_tensor) ** 2)
sst = torch.sum((y_test_tensor - torch.mean(y_test_tensor)) ** 2)
r2 = 1 - ssr / sst
print("R2 Score:", r2.item())

#%%
time_index = X_test.index[time_steps:]

# 创建一个Pandas Series来表示预测结果，使用提取的时间索引
predictions_series = pd.Series(predictions.flatten(), index=time_index)

#%%
n = 1000
plt.figure(figsize=(10, 6))
plt.plot(y_test[time_steps:n], label='Actual', color='red')
plt.plot(predictions_series[:n-time_steps], label='Predicted', color='blue')
plt.axhline(y=0, color='gray', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Avg Price change')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()



