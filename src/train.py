from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from src.data_loader import load_and_split_data

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# def main():
#     # 1. 获取训练集和测试集
#     X_train, X_test, y_train, y_test = load_and_split_data()
#
#     # 2. 创建模型
#     model = LogisticRegression(max_iter=1000)
#
#     # 3. 训练模型
#     model.fit(X_train, y_train)
#
#     # 4. 预测测试集
#     y_pred = model.predict(X_test)
#
#     # 5. 提取前10个预测标签和真实标签
#     # pred_first10 = y_pred[:10]
#     # true_first10 = y_test.iloc[:10].to_numpy()
#
#     # # 6. 查看预测结果
#     # print("前10个预测标签:", pred_first10)
#     # print("前10个真实标签:", true_first10)
#     #
#     # # 7. 逐个比较预测是否正确
#     # print("\n预测结果对比：")
#     # for i,(true_lablel, pred_lablel) in enumerate(zip(true_first10, pred_first10), start=1):
#     #     is_correct = true_lablel == pred_lablel
#     #     print(
#     #         f"样本{i}:"
#     #         f"真实值={true_lablel},"
#     #         f"预测值={pred_lablel}",
#     #         f"是否正确={is_correct}"
#     #     )
#
#     true_test = y_test.iloc[:].to_numpy()
#     print("预测标签:", y_pred)
#     print("真实标签:", true_test)
#
#     for i, (true_lablel, pred_lablel) in enumerate(zip(true_test, y_pred),start=1):
#         is_correct = true_lablel == pred_lablel
#         print(
#             f"样本{i}:"
#             f"真实值={true_lablel},"
#             f"预测值={pred_lablel}",
#             f"是否正确={is_correct}"
#         )
#
#     # ==========================================
#     # Day 5：模型评估 - Accuracy
#     # ==========================================
#
#     # 从 sklearn.metrics 导入 accuracy_score
#
#     # 使用 accuracy_score() 计算全部测试集的准确率
#     # 提示：需要 y_test 和 y_pred
#     accuracy = accuracy_score(y_test, y_pred)
#
#     # 将准确率打印出来
#     print("Accuracy:", accuracy)
#
#     # count = 0
#     # for i, value in enumerate(y_pred):
#     #     if value == true_test[i]:
#     #         count += 1
#     #
#     # accuracy1 = count / len(y_pred)
#     # print("Accuracy1:", accuracy1)
#
#     count = 0
#
#     for true_label, pred_label in zip(true_test, y_pred):
#         if true_label == pred_label:
#             count += 1
#     accuracy1 = count / len(y_pred)
#     print("Accuracy1:", accuracy1)
#
#     print("Scikit-learn Accuracy:", accuracy)
#     print("Manual Accuracy:", accuracy1)
#     print("两种结果是否一致:", accuracy == accuracy1)
#
#     # ==========================================
#     # Day 5 第二阶段：Confusion Matrix
#     # ==========================================
#     # 从 sklearn.metrics 导入 confusion_matrix
#     # 注意：import通常放在文件顶部

#     # 调用 confusion_matrix()
#     # 输入：真实标签 y_test、预测标签 y_pred
#     # 保存到变量 cm
#     cm = confusion_matrix(y_test, y_pred)

#     # 打印混淆矩阵 cm
#     print(cm)

#     # 使用 cm.shape 查看矩阵的形状
#     # 思考：为什么应该是 (3, 3)？
#     print(cm.shape)



def main():
    # ==============================
    # 1. 加载训练集和测试集
    # ==============================
    X_train, X_test, y_train, y_test = load_and_split_data()

    # ==============================
    # 2. 创建并训练模型
    # ==============================
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # ==============================
    # 3. 预测测试集
    # ==============================
    y_pred = model.predict(X_test)

    # # ==============================
    # # 4. Accuracy：准确率
    # # ==============================
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy:.4f}")
    #
    # # ==============================
    # # 5. Confusion Matrix：混淆矩阵
    # # ==============================
    # cm = confusion_matrix(y_test, y_pred)
    #
    # print("\nConfusion Matrix:")
    # print(cm)
    #
    # # ==============================
    # # 6. Precision / Recall / F1
    # # ==============================
    # class_names = load_iris().target_names
    #
    # report = classification_report(
    #     y_test,
    #     y_pred,
    #     target_names=class_names,
    #     digits=4
    # )
    #
    # print("\nClassification Report:")
    # print(report)

    # ==========================================
    # Day 6：标准化 + Pipeline
    # ==========================================

    # 在文件顶部导入：
    # StandardScaler
    # make_pipeline

    # 创建一个Pipeline，包含：
    # StandardScaler()
    # LogisticRegression(max_iter=1000)



    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    # 使用pipeline.fit()训练
    # 提示：传入X_train和y_train
    '''
     pipeline.fit(X_train, y_train)执行步骤：
     X_train + y_train： 训练特征与对应标签输入
     第一步：StandardScaler.fit(X_train): 学习训练集每个特征的均值、标准差
     第二步：StandardScaler.transform(X_train): 把训练特征转换成标准化后的数值
     第三步：LogisticRegression.fit(): 使用标准化后的训练特征和y_train学习分类规律
     '''
    pipeline.fit(X_train, y_train)

    # 使用pipeline.predict()预测X_test
    # 把结果保存到pipeline_pred
    '''
     pipeline_pred = pipeline.predict(X_test)执行步骤：
     X_test：原始测试特征
     第一步：StandardScaler.transform(X_test): 使用训练阶段学到的均值和标准差
     第二步：LogisticRegression.predict(): 使用训练好的分类器进行预测
     '''
    pipeline_pred = pipeline.predict(X_test)


    # 使用accuracy_score()计算新模型准确率
    pipeline_accuracy = accuracy_score(y_test, pipeline_pred)

    # 打印原模型Accuracy与Pipeline模型Accuracy
    # 观察两者是否相同
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Pipline_Accuracy: {pipeline_accuracy:.4f}")

    # ==========================================
    # Day 6 第二阶段：验证标准化器
    # ==========================================

    # TODO 1：
    # 从pipeline中获取StandardScaler对象
    scaler = pipeline.named_steps["standardscaler"]

    # TODO 2：
    # 打印标准化器学习到的均值
    print("Scaler均值:", scaler.mean_)

    # TODO 3：
    # 使用Pandas的mean()计算训练集每列的均值
    # 提示：X_train.mean()
    train_mean = X_train.mean()

    # TODO 4：
    # 使用Pandas的mean()计算测试集每列的均值
    # 提示：X_test.mean()
    test_mean = X_test.mean()
    # TODO 5：
    # 对比三组均值：
    # ① scaler.mean_
    # ② X_train.mean()
    # ③ X_test.mean()
    print("Train mean:", train_mean.to_numpy())
    print("Test mean:", test_mean.to_numpy())

    scaler = pipeline.named_steps["standardscaler"]


if __name__ == "__main__":
    main()
