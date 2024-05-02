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
import torch.nn.functional as F
import shap

#%%

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_length, hidden_dim]
        attention_scores = self.linear(lstm_output)  # [batch_size, seq_length, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # 对每个序列进行softmax
        context_vector = attention_weights * lstm_output  # [batch_size, seq_length, hidden_dim]
        context_vector = torch.sum(context_vector, dim=1)  # [batch_size, hidden_dim]
        return context_vector, attention_weights
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)  # 输出层直接连接到分类数量

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)  # [batch_size, seq_length, hidden_dim]
        lstm_out = self.dropout(lstm_out)

        # 注意力层
        attn_out, attn_weights = self.attention(lstm_out)  # 使用注意力层

        # 全连接层
        out = self.fc(attn_out)  # [batch_size, output_dim]
        return out, attn_weights  # 返回输出和注意力权重
def create_sequences(input_data, target_data, sequence_length):
    sequences = []
    target_sequences = []
    for i in range(len(input_data) - sequence_length):
        sequences.append(input_data[i:i+sequence_length])
        target_sequences.append(target_data[i+sequence_length])
    return np.array(sequences), np.array(target_sequences)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_accuracy = 0.0  # 初始化最佳准确率
    best_model = None  # 初始化最佳模型权重
    best_epoch = 0  # 初始化最佳模型的训练轮次

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # 由于模型修改为返回一个元组，我们这里仅使用输出
            y_pred, _ = model(X_batch)  # 获取模型的输出和注意力权重
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 计算平均训练损失
        train_loss /= len(train_loader)

        # 进入评估模式
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                y_val_pred, _ = model(X_val_batch)  # 同样，仅使用输出
                test_loss += criterion(y_val_pred, y_val_batch).item()
                _, predicted = torch.max(y_val_pred.data, 1)
                total += y_val_batch.size(0)
                correct += (predicted == y_val_batch).sum().item()

        # 计算平均验证损失和准确率
        test_loss /= len(val_loader)
        test_accuracy = correct / total

        # 更新最佳模型（如果适用）
        if test_accuracy > best_accuracy:
            best_epoch = epoch
            best_accuracy = test_accuracy
            best_model = model.state_dict()
            torch.save(best_model, 'model/best_model.pth')

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, '
              f'Best Epoch: {best_epoch+1}')

    model.load_state_dict(torch.load('model/best_model.pth'))

# 预测
def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_test_pred,_ = model(X_batch)
            predictions.append(y_test_pred.numpy())
    return np.concatenate(predictions, axis=0)

def mark_label(df,k,thresholds):
    #0:stay,1:up，2:down
    # df['m_minus'] = df['avg_price'].rolling(window=k).mean()
    df['m_plus'] = df['avg_price'].shift(-1).rolling(window=k, min_periods=1).mean().shift(-k+1)
    df['l_t'] = (df['m_plus'] - df['avg_price']) / df['avg_price']
    df['label'] = 0
    df.loc[df['l_t'] > thresholds, 'label'] = 1
    df.loc[df['l_t'] < -thresholds, 'label'] = 2

    return df

#%%
from pandas import Timedelta
df = pd.read_csv('process_data/total_lob_1.csv')
df = df.dropna()
# 将time_window列（秒数）转换为timedelta类型
df['date'] = pd.to_datetime(df['date'])
# 将时间窗口转换为timedelta（时间窗口以秒为单位），并设置每天的起始时间为8:00
df['datetime'] = df['date'] + pd.to_timedelta('8 hours') + pd.to_timedelta(df['time_window'], unit='s')
# 设置新的日期时间为索引
df.set_index('datetime', inplace=True)


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
input_dim = X_train.shape[2]
hidden_dim = 100 # 隐藏层维度
num_layers = 5 # LSTM层的数量
output_dim =  3# 输出维度
dropout_prob = 0.2

model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim,dropout_prob)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 用于多分类问题
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#%%
# 训练模型
num_epochs = 20
train_model(model, train_loader,test_loader,criterion, optimizer, num_epochs)

#%%
# 测试模型性能# 此函数应该返回模型在测试集上的预测
#model.load_state_dict(torch.load('model/best_model(2).pth'))
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


#%%
df_index_test = df_index[train_size:]
test_timestamps = df_index_test[sequence_length :]
avg_price = df['avg_price'].values[train_size+sequence_length:]
df_result = pd.DataFrame({
    'Actual': y_test_tensor.numpy(),
    'Forecast': predicted_classes.numpy(),
    'avg_price':avg_price
},index=test_timestamps)

df_result.to_csv('test_predictions_comparison.csv')