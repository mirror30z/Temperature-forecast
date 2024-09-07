import torch
from utils.data_loader import load_data
from model.model import create_model
from train import train_model
from eval import evaluate_model
from plot import plot_predictions
import os
def main():
    # 1加载数据 替换为你的目录
    #input_features, labels, dates, feature_list = load_data(r'E:\teach\tang\pytorch\Temperature forecast\data\temps.csv')
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'data', 'temps.csv')
    # 加载数据
    input_features, labels, dates, feature_list = load_data(data_path)

    # 获取输入特征的维度
    input_size = input_features.shape[1]

    # 创建模型
    model = create_model(input_size)

    # 训练模型
    model = train_model(model, input_features, labels)

    # 使用训练好的模型进行预测
    predictions = evaluate_model(model, input_features)

    # 绘制预测结果与真实值
    plot_predictions(dates, labels, predictions, save_path='prediction_results.png')

if __name__ == '__main__':
    main()
