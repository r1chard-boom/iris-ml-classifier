from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from src.data_loader import load_and_split_data
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

    # ==============================
    # 4. Accuracy：准确率
    # ==============================
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # ==============================
    # 5. Confusion Matrix：混淆矩阵
    # ==============================
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # ==============================
    # 6. Precision / Recall / F1
    # ==============================
    class_names = load_iris().target_names

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4
    )

    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
