# 神经网络回归——气温预测

该项目使用 PyTorch 实现了一个深度学习模型，用于预测每天的最高气温。项目采用模块化设计，包含数据预处理、模型定义、训练、评估和结果可视化模块。最终结果会以图表的形式展示，并保存为图片文件。

## 项目结构

项目的文件结构如下：

```
Temperature forecast/
│
├── data/                     # 包含数据集的目录
│   └── temps.csv             # CSV 文件，包含气温数据
│
├── model/                    # 模型定义的目录
│   └── model.py              # 神经网络模型定义
│
├── utils/                    # 工具目录，包含数据加载等功能
│   └── data_loader.py        # 数据加载与预处理脚本
│
├── all.py                    # （可选）包含所有功能可直接运行
├── eval.py                   # 模型评估脚本
├── main.py                   # 主程序脚本，协调整个流程
├── plot.py                   # 结果绘图脚本
├── train.py                  # 模型训练脚本
├── prediction_results.png    # 示例输出图片，包含预测与实际温度对比
├── README.md                 # 项目说明文档

```

## 环境依赖

要运行此项目，您需要安装以下依赖项：

```bash
pip install numpy pandas matplotlib torch scikit-learn
```

## 数据集

![数据](E:\teach\tang_system_class\04pytorch\images\气温预测\数据.png)

数据集 (`temps.csv`) 包含每日的天气信息，包含以下列：

- `year`: 观测年份
- `month`: 观测月份
- `day`: 观测日
- `temp_1`: 前一天的最高温度
- `temp_2`: 前两天的最高温度
- `actual`: 当天的实际最高温度（目标变量）
- friend：朋友猜测的可能值

## Run

**1.克隆仓库或者直接下载：**

```python
git clone https://github.com/your-username/temperature-forecast.git
cd temperature-forecast
```

**2.放置数据集**：确保 `temps.csv` 文件位于 `data/` 目录下。

**3.运行主脚本**： 要加载数据、训练模型、评估并绘制结果，请运行 `main.py` 脚本：

```python
python main.py
```

## Result

真实值和预测值结果展示

![prediction_results](E:\teach\tang_system_class\04pytorch\Temperature forecast\prediction_results.png)