
import joblib
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris


def main():
    # ==============================
    # 1. 获取项目根目录和模型路径
    # ==============================
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    model_path = PROJECT_ROOT / "models" / "iris_pipeline.joblib"

    # ==============================
    # 2. 加载已经训练好的Pipeline
    # ==============================
    loaded_pipeline = joblib.load(model_path)
    print("模型加载成功！")

    # ==============================
    # 3. 获取特征名称和类别名称
    # ==============================
    iris = load_iris()

    # ==============================
    # 4. 创建一条新的预测样本
    # ==============================
    new_sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=iris.feature_names
    )

    print("\n新样本数据：")
    print(new_sample)

    # ==============================
    # 5. 使用加载的模型进行预测
    # ==============================
    predicted_id = loaded_pipeline.predict(new_sample)[0]

    # ==============================
    # 6. 将类别编号转换为花朵名称
    # ==============================
    predicted_name = iris.target_names[predicted_id]

    # ==============================
    # 7. 输出预测结果
    # ==============================
    print("\n预测结果：")
    print("预测类别编号:", predicted_id)
    print("预测花朵名称:", predicted_name)


if __name__ == "__main__":
    main()
