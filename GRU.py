import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, \
    accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


# 定义GRU模型类
class GRU_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hidden = self.gru(x)
        output = self.fc(hidden[-1])  # 使用最后一个时间步的隐藏状态进行分类
        return output

def create_sequences(input_data, target_data, sequence_length):
    sequences = []
    target_sequences = []
    for i in range(len(input_data) - sequence_length):
        sequences.append(input_data[i:i+sequence_length])
        target_sequences.append(target_data[i+sequence_length])
    return np.array(sequences), np.array(target_sequences)


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    # 将模型设置为训练模式
    model.train()
    best_accuracy = 0.0  # 初始化最佳准确率
    best_model = None  # 初始化最佳模型权重
    best_epoch = 0  # 初始化最佳模型的训练轮次

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            running_loss += loss.item()



        # 每个epoch结束后，测试模型
        test_accuracy,test_loss,_=test_model(model, test_loader, criterion)
        train_loss = running_loss / len(train_loader)

        if test_accuracy > best_accuracy:
            best_epoch = epoch
            best_accuracy = test_accuracy
            best_model = model.state_dict()
            torch.save(best_model, 'model/best_model(GRU).pth')

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, '
              f'Best Epoch: {best_epoch+1}')




def test_model(model, test_loader, criterion):
    # 将模型设置为评估模式
    model.eval()

    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():  # 不计算梯度
        for inputs, labels in test_loader:
            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    loss = running_loss / len(test_loader)
    return accuracy, loss, predicted

#%%

df = pd.read_csv('process_data/UoB_Set01_2025-01-02LOBs.csv')
df = df.dropna()
# df = df.iloc[:5000]
start_date = pd.to_datetime('2025-01-02 08:00:00')
df['actual_datetime'] = start_date + pd.to_timedelta(df['time_window'], unit='s')
df.set_index('actual_datetime', inplace=True)
df_index = df.index
feature=df[['avg_price','avg_price_change', 'bid_level_diff', 'ask_level_diff',
             'bid_ask_depth_diff']]

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
# 创建模型实例
input_dim = X_train.shape[2]  # 输入维度
hidden_dim = 64  # 隐藏层维度
output_dim = 3  # 输出维度
# 创建模型实例
model = GRU_Model(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#%%
num_epochs = 20  # 可根据需要调整
train_model(model, train_loader, test_loader,criterion, optimizer, num_epochs=num_epochs)
#%%
model.load_state_dict(torch.load('model/best_model(GRU).pth'))
# 测试模型

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch) # 假设模型返回元组，预测值在第一位
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(y_batch.tolist())
        y_pred.extend(predicted.tolist())
cm=confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(10, 7))  # 设置图形大小
cax = ax.matshow(cm, cmap='Blues')  # 创建矩阵图
fig.colorbar(cax)  # 添加色彩条

for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, val, ha='center', va='center',
            color='white' if cm[i, j] > cm.max()/2 else 'black',
            )

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
plt.savefig('confusion_matrix(GRU).png')
plt.show()

accuracy = accuracy_score(y_true, y_pred)
print(classification_report(y_true, y_pred))
# 计算精确度，召回率，和F1分数。这里我们使用"macro"平均，它对每个类别的指标计算平均，不考虑类别的样本量
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')