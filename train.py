
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import torch
import numpy as np

def train_model(model, input_features, labels, batch_size=64, epochs=2000, lr=0.001):
    # 定义损失函数和优化器
    cost = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for i in range(epochs):
        batch_loss = []
        # Mini-Batch 训练
        for start in range(0, len(input_features), batch_size):
            end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
            xx = torch.tensor(input_features[start:end], dtype=torch.float32, requires_grad=True)
            yy = torch.tensor(labels[start:end], dtype=torch.float32, requires_grad=True)

            # 前向传播
            prediction = model(xx)
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

    return model
