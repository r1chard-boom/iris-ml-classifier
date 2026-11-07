from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_and_split_data
from src.evaluate import evaluate_model



def main():

    # 1. 加载训练集和测试集
    X_train, X_test, y_train, y_test = load_and_split_data()


    # # 2. 创建并训练模型
    # model = LogisticRegression(max_iter=1000)
    # model.fit(X_train, y_train)
    #
    # # 3. 预测测试集
    # y_pred = model.predict(X_test)

    # 4.标准化 + Pipeline
    # 创建一个Pipeline，包含：
    # StandardScaler()
    # LogisticRegression(max_iter=1000)
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    # 使用pipeline.fit()训练
    # 提示：传入X_train和y_train

     # pipeline.fit(X_train, y_train)执行步骤：
     # X_train + y_train： 训练特征与对应标签输入
     # 第一步：StandardScaler.fit(X_train): 学习训练集每个特征的均值、标准差
     # 第二步：StandardScaler.transform(X_train): 把训练特征转换成标准化后的数值
     # 第三步：LogisticRegression.fit(): 使用标准化后的训练特征和y_train学习分类规律

    pipeline.fit(X_train, y_train)

    # 5.使用pipeline.predict()预测X_test
    # 把结果保存到pipeline_pred

     # pipeline_pred = pipeline.predict(X_test)执行步骤：
     # X_test：原始测试特征
     # 第一步：StandardScaler.transform(X_test): 使用训练阶段学到的均值和标准差
     # 第二步：LogisticRegression.predict(): 使用训练好的分类器进行预测

    pipeline_pred = pipeline.predict(X_test)

    class_names = load_iris().target_names

    evaluate_model(
        y_test,
        pipeline_pred,
        class_names
    )

    # 6. 保存训练好的Pipeline

    # 创建一个模型文件路径
    # models/iris_pipeline.joblib
    # 1) 获取项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    print(PROJECT_ROOT)

    # 2) 构建模型保存路径
    model_path = PROJECT_ROOT / "models" / "iris_pipeline.joblib"

    # 确保models文件夹存在
    # 提示：model_path.parent.mkdir(...)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用joblib.dump()保存训练好的pipeline
    # 需要传入两个参数：
    # ① 已训练好的pipeline
    # ② 保存位置model_path
    joblib.dump(pipeline, model_path)

    # 打印模型保存路径，方便检查
    print("model saved: {}".format(model_path))



if __name__ == "__main__":
    main()
