import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings
import datetime
from sklearn import preprocessing
import os

# 1加载数据  替换为你的目录
# features = pd.read_csv(r'E:\teach\tang\pytorch\Temperature forecast\data\temps.csv')
# 2加载数据  确保当前项目data中有数据集
current_dir = os.path.dirname(__file__)
# 构建相对路径
data_path = os.path.join(current_dir, 'data', 'temps.csv')
# 读取数据
features = pd.read_csv(data_path)
# 处理时间数据
years = features['year']
months = features['month']
days = features['day']

# 将日期转换为 datetime 格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 独热编码
features = pd.get_dummies(features)

# 提取标签列 'actual'
labels = np.array(features['actual'])

# 在特征中去掉标签列
features = features.drop('actual', axis=1)

# 保存特征列名
feature_list = list(features.columns)

# 转换为 NumPy 数组
features = np.array(features)

# 标准化特征
input_features = preprocessing.StandardScaler().fit_transform(features)

# 定义网络结构参数
input_size = input_features.shape[1]
hidden_size = 128
output_size = 1
batch_size = 16

# 定义神经网络
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size),
)

# 损失函数和优化器
cost = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

# 训练网络
losses = []
for i in range(1000):
    batch_loss = []
    # Mini-Batch 训练
    for start in range(0, len(input_features), batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype=torch.float32, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float32, requires_grad=True)

        # 前向传播
        prediction = my_nn(xx)
        loss = cost(prediction, yy.view(-1, 1))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # 记录损失
        batch_loss.append(loss.data.numpy())

    # 每 100 次迭代打印一次损失
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(f'Epoch {i}, Loss: {np.mean(batch_loss)}')

# 模型预测
x = torch.tensor(input_features, dtype=torch.float32)
predict = my_nn(x).data.numpy()

# 创建一个 DataFrame 来存储日期和标签（真实值）
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})

# 创建一个 DataFrame 来存储日期和模型预测值
test_dates = dates
predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predict.reshape(-1)})

# 绘制真实值和预测值的对比图
plt.plot(true_data['date'], true_data['actual'], 'b-', label='Actual')

# 绘制预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='Prediction')

# 调整图表
plt.xticks(rotation=60)
plt.legend()

# 设置标题和坐标轴标签
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')

# 显示图表
plt.tight_layout()
plt.show()
