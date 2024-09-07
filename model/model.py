import torch

def create_model(input_size, hidden_size=128, output_size=1):
    # 定义神经网络
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_size, output_size),
    )
    return model

