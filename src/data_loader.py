import pandas
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# Data Loading
iris = load_iris(as_frame=True)

# Data Understanding
print('iris的数据类型:', type(iris.data))
print("表格前5行数据:", iris.data.head())

print("数据集的形状:\n", iris.data.shape)
print("四个特征名称:\n", iris.feature_names)
print("三种花的类别名称:\n", iris.target_names)
print("前10行特征(X):\n", iris.data.head(10))
print("前10行标签(Y):\n", iris.target.head(10))

# Data Exploring
print("特征的统计分布:\n", iris.data.describe())
print("检查数据是否缺失:\n", iris.data.isnull())
print("每个特征的缺失值数量:\n", iris.data.isnull().sum())
print("========================================")
print(iris.target.value_counts())
print("========================================")

# ==========================================
# Day 3：训练集与测试集划分
# ==========================================


# 获取特征矩阵x和标签y
X = iris.data
y = iris.target

# TODO 5：调用train_test_split()
# 要求：
# ① 同时传入 X 和 Y
# ② 测试集比例为20%
# ③ random_state设置为42
# ④ 按标签y进行分层划分
# ⑤ 用四个变量接收返回值
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TODO 6：打印训练特征的shape
print(X_train.shape)
# TODO 7：打印测试特征的shape
print(X_test.shape)
# TODO 8：打印训练标签的shape
print(y_train.shape)
# TODO 9：打印测试标签的shape
print(y_test.shape)

# TODO 10：使用value_counts()
# 查看训练集中每个类别的样本数
y_train_counts = y_train.value_counts()
print(y_train_counts)


print("=====================================================================")
# TODO 11：使用value_counts() value_counts():不同类别出现次数
# 查看测试集中每个类别的样本数
y_test_counts = y_test.value_counts()
print(y_test_counts)

# test: <class 'pandas.core.series.Series'>
# print(type(y_test))
# a = [1, 2, 2, 1, 1]
# tmp = pandas.Series(a)
# print(tmp.value_counts())