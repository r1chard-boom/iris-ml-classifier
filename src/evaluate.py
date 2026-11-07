
# src/evaluate.py

# TODO 1：
# 从sklearn.metrics导入三个函数：
# accuracy_score
# confusion_matrix
# classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def evaluate_model(y_true, y_pred, class_names):

    # TODO 2：
    # 使用accuracy_score计算准确率
    accuracy = accuracy_score(y_true, y_pred)


    # TODO 3：
    # 使用confusion_matrix计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # TODO 4：
    # 使用classification_report生成分类报告
    # target_names设置为class_names
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits = 4
    )

    # TODO 5：
    # 打印三个评估结果
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)