from sklearn.datasets import load_iris
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