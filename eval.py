import torch


def evaluate_model(model, input_features):
    # 将数据转为 tensor
    x = torch.tensor(input_features, dtype=torch.float32)
    # 前向传播，获取预测结果
    predictions = model(x).data.numpy()
    return predictions


