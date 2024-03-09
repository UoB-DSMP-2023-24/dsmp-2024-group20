#%%
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
import seaborn as sns

seed_value = 1  # 你可以选择任何喜欢的种子值
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#%%
# 创建一个新的figure和一个或多个subplot
fig, ax = plt.subplots(figsize=(14, 7))
df = pd.read_csv('process_data/UoB_Set01_2025-01-02LOBs.csv')
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
#%%
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)  # 输出层直接连接到分类数量

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # 取序列最后一步的输出
        return out
def create_sequences(input_data, target_data, sequence_length):
    sequences = []
    target_sequences = []
    for i in range(len(input_data) - sequence_length):
        sequences.append(input_data[i:i+sequence_length])
        target_sequences.append(target_data[i+sequence_length])
    return np.array(sequences), np.array(target_sequences)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_loss = float('inf')  # 初始化最佳损失为无穷大
    best_model = None  # 初始化最佳模型
    best_epoch = 0  # 初始化最佳模型的训练轮次
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # 计算平均训练损失
        train_loss /= len(train_loader)
        # 验证过程
        model.eval()  # 将模型设置为评估模式
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # 在验证过程中不计算梯度
            for X_val_batch, y_val_batch in val_loader:
                y_val_pred = model(X_val_batch)
                test_loss += criterion(y_val_pred, y_val_batch).item()
                _, predicted = torch.max(y_val_pred.data, 1)
                total += y_val_batch.size(0)
                correct += (predicted == y_val_batch).sum().item()

        # 计算平均验证损失和准确率
        test_loss /= len(val_loader)
        test_accuracy = correct / total

        if test_loss < best_loss:
            best_epoch = epoch
            best_loss = test_loss
            best_model = model.state_dict()  # 保存最佳模型的状态字典
            # 保存模型状态字典
            torch.save(best_model, 'model/best_model.pth')

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, 'f'Test Accuracy: {test_accuracy:.4f}, '
              f'Best Epoch: {best_epoch+1}')

# 预测
def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_test_pred = model(X_batch)
            predictions.append(y_test_pred.numpy())
    return np.concatenate(predictions, axis=0)


#%%

df = pd.read_csv('process_data/UoB_Set01_2025-01-02LOBs.csv')
df = df.dropna()
# df = df.iloc[:5000]
start_date = pd.to_datetime('2025-01-02 08:00:00')
df['actual_datetime'] = start_date + pd.to_timedelta(df['time_window'], unit='s')
df.set_index('actual_datetime', inplace=True)
feature=df[['max_bid', 'min_ask', 'avg_price','avg_price_change',
            'bid_level_diff', 'ask_level_diff','bid_ask_depth_diff']]
target=df['label']

scaler = StandardScaler()

# 对特征进行标准化
scaled_features = scaler.fit_transform(feature)
sequence_length = 10  # 可以根据需要调整这个值
X, y = create_sequences(scaled_features, target.values, sequence_length)
# 按顺序划分数据集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# test_size = int(len(X_temp) * 0.2)
# X_val, X_test = X_temp[:test_size], X_temp[test_size:]
# y_val, y_test = y_temp[:test_size], y_temp[test_size:]

# X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val, dtype=torch.long)

X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


batch_size = 64

train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
# val_data = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
# val_loader = DataLoader(val_data, batch_size=64)


#%%
input_dim = X_train.shape[2]
hidden_dim = 100 # 隐藏层维度
num_layers = 5 # LSTM层的数量
output_dim =  3# 输出维度
dropout_prob = 0.2

model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim,dropout_prob)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 用于多分类问题
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10
train_model(model, train_loader,test_loader,criterion, optimizer, num_epochs)

#%%
# 测试模型性能# 此函数应该返回模型在测试集上的预测
model.load_state_dict(torch.load('model/best_model.pth'))
predictions = predict(model, test_loader)

# 将输出的概率转换为类别索引
_, predicted_classes = torch.max(torch.tensor(predictions), 1)


# 计算准确率
correct_predictions = (predicted_classes == y_test_tensor).sum().item()
accuracy = correct_predictions / y_test_tensor.size(0)
print(f'Accuracy: {accuracy:.4f}')

cm = confusion_matrix(y_test_tensor, predicted_classes.numpy())

fig, ax = plt.subplots()  # 设置足够大的图形尺寸
cax = ax.matshow(cm, cmap="Blues")  # 选择色彩图谱
plt.colorbar(cax)  # 显示色条

# 在每个单元格中添加注释文本
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, val, ha='center', va='center',
            color='white' if cm[i, j] > cm.max()/2 else 'black',
            )  # 根据需要调整字体大小

# 设置轴
ax.set_xticks(np.arange(cm.shape[1]))
ax.set_yticks(np.arange(cm.shape[0]))
ax.set_xticklabels(['stay', 'up', 'down'])
ax.set_yticklabels(['stay', 'up', 'down'])
ax.xaxis.set_label_position('bottom')
ax.xaxis.tick_bottom()  # 将x轴标签移到下方

# 调整标签字体大小
ax.tick_params(axis='both', which='major', labelsize=10)

# 调整子图布局以防止标签被挤出视图
plt.subplots_adjust(bottom=0.2, top=0.9)  # 可以根据需要调整这些参数

plt.xlabel('Predicted', labelpad=20)
plt.ylabel('Truth', labelpad=10)
plt.savefig('confusion_matrix.png')
plt.show()

print(classification_report(y_test_tensor, predicted_classes))

precision = precision_score(y_test_tensor, predicted_classes.numpy(), average='macro')
recall = recall_score(y_test_tensor, predicted_classes.numpy(), average='macro')
f1 = f1_score(y_test_tensor, predicted_classes.numpy(), average='macro')

print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")

