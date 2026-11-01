
from sklearn.linear_model import LogisticRegression
from src.data_loader import load_and_split_data


def main():
    # 1. 获取训练集和测试集
    X_train, X_test, y_train, y_test = load_and_split_data()

    # 2. 创建模型
    model = LogisticRegression(max_iter=1000)

    # 3. 训练模型
    model.fit(X_train, y_train)

    # 4. 预测测试集
    y_pred = model.predict(X_test)

    # 5. 提取前10个预测标签和真实标签
    # pred_first10 = y_pred[:10]
    # true_first10 = y_test.iloc[:10].to_numpy()

    # # 6. 查看预测结果
    # print("前10个预测标签:", pred_first10)
    # print("前10个真实标签:", true_first10)
    #
    # # 7. 逐个比较预测是否正确
    # print("\n预测结果对比：")
    # for i,(true_lablel, pred_lablel) in enumerate(zip(true_first10, pred_first10), start=1):
    #     is_correct = true_lablel == pred_lablel
    #     print(
    #         f"样本{i}:"
    #         f"真实值={true_lablel},"
    #         f"预测值={pred_lablel}",
    #         f"是否正确={is_correct}"
    #     )

    true_test = y_test.iloc[:].to_numpy()
    print("预测标签:", y_pred)
    print("真实标签:", true_test)

    for i, (true_lablel, pred_lablel) in enumerate(zip(y_pred,true_test),start=1):
        is_correct = true_lablel == pred_lablel
        print(
            f"样本{i}:"
            f"真实值={true_lablel},"
            f"预测值={pred_lablel}",
            f"是否正确={is_correct}"
        )








if __name__ == "__main__":
    main()