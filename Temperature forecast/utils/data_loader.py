import pandas as pd
import numpy as np
from sklearn import preprocessing
import datetime

def load_data(file_path):
    # 加载数据
    features = pd.read_csv(file_path)

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
    # 转换为 NumPy 数组并标准化
    input_features = preprocessing.StandardScaler().fit_transform(np.array(features))

    return input_features, labels, dates, feature_list


