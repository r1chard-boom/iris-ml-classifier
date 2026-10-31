# from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt
#
# # 1.加载数据集
# iris = load_iris(as_frame=True)
#
# # 2.获取特征数据
# X = iris.data
#
# # 3.获取标签
# labels = iris.target
# print(iris.target_names)
# # 4.提取花瓣长度和花瓣宽度
# x = X["petal length (cm)"]
# y = X["petal width (cm)"]
#
# # mask0 = labels == 0
# # plt.scatter(x[mask0], y[mask0], label='setosa')
# # # 第1类：Versicolor
# # mask1 = labels == 1
# # plt.scatter(x[mask1], y[mask1], label='versicolor')
# #
# # # 第2类：Virginica
# # mask2 = labels == 2
# # plt.scatter(x[mask2], y[mask2], label='virginica')
# #
# # plt.legend()
# # plt.show()
# # # 5: 使用plt.scatter()绘制散点图
# # plt.scatter(x, y, c=labels, cmap='viridis')
# #
# # # 6: 设置横轴名称 Petal Length (cm)
# # plt.xlabel("petal length (cm)")
# # # 7: 设置纵轴名称 Petal Width (cm)
# # plt.ylabel("petal width (cm)")
# #
# # # 8: 显示图表
# # plt.show()
#
# # 遍历三种iris
# for i, name in enumerate(iris.target_names):
#
#     # 筛选当前类型的数据
#     mask = labels == i
#
#     # 使用plt.scatter()
#     # 只绘制x[mask]和y[mask]
#     # 并设置label=name
#     plt.scatter(x[mask], y[mask], label=name)
#
# # 添加横轴名称
# plt.xlabel("petal length (cm)")
# # 添加纵轴名称
# plt.ylabel("petal width (cm)")
# # 调用plt.legend()显示图例
# plt.legend()
# # 调用plt.show()
# plt.show()

# ============================================================
# Day 3：Iris 数据可视化独立复现练习
# 目标：绘制三种鸢尾花的花瓣长度与宽度散点图
# 要求：不同类别使用不同颜色，并显示正确的图例
# ============================================================


# ==================== 第一部分：导入库 ====================
# TODO 1：从 sklearn.datasets 导入鸢尾花数据加载函数
from sklearn.datasets import load_iris
# TODO 2：导入 matplotlib.pyplot，并使用 plt 作为别名
import matplotlib.pyplot as plt

# ==================== 第二部分：加载数据 ====================

# TODO 3：加载鸢尾花数据集，要求特征以 Pandas DataFrame 形式返回
iris = load_iris(as_frame=True)
# TODO 4：提取特征矩阵，保存为变量 X
X = iris.data
# TODO 5：提取类别标签，保存为变量 labels
labels = iris.target #target/label
# print(labels)
# ==================== 第三部分：提取绘图数据 ====================

# TODO 6：从 X 中提取 petal length (cm)，保存为变量 x
x = X["petal length (cm)"]
# print(x)
# print(x.shape)
# TODO 7：从 X 中提取 petal width (cm)，保存为变量 y
y = X["petal width (cm)"]

# ==================== 第四部分：绘制分类散点图 ====================

# TODO 8：使用 for 循环遍历三种花的名称
# 提示：需要同时得到类别编号 i 和类别名称 name
for i, name in enumerate(iris.target_names):

    # TODO 9：创建布尔索引 mask
    # 要求：筛选出标签等于当前类别编号 i 的样本
    mask = labels == i
    # TODO 10：绘制当前类别的散点图
    # 要求：
    # ① 使用 x[mask] 作为横轴数据
    # ② 使用 y[mask] 作为纵轴数据
    # ③ 使用当前类别名称 name 作为 label
    plt.scatter(x[mask], y[mask], label=name)

# ==================== 第五部分：完善图表 ====================

# TODO 11：设置横轴名称为 Petal Length (cm)
plt.xlabel("petal length (cm)")
# TODO 12：设置纵轴名称为 Petal Width (cm)
plt.ylabel("petal width (cm)")
# TODO 13：设置图表标题为 Iris Flower Classification
plt.title("Iris Flower Classification")
# TODO 14：显示三个类别的图例
plt.legend()
# TODO 15：显示最终图表
plt.show()
