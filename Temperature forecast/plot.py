import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions(dates, true_values, predictions,save_path = 'prediction_plot.png'):
    predictions = predictions.reshape(-1)
    # 创建一个 DataFrame 来存储日期和标签（真实值）
    true_data = pd.DataFrame(data={'date': dates, 'actual': true_values})

    # 创建一个 DataFrame 来存储日期和模型预测值
    predictions_data = pd.DataFrame(data={'date': dates, 'prediction': predictions})

    # 绘制真实值
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

    plt.savefig(save_path)
    plt.show()
