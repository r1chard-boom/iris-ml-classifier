from sklearn.datasets import load_iris

project_name = 'Iris ML Classifier'

features = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width'
]

def show_project_info():
    print('Project Name:', project_name)
    print('Features number:', len(features))
    for feature in features:
        print(feature)
    print()

if __name__ == '__main__':
    show_project_info()
    iris = load_iris()
    print("Feature names：", iris.feature_names)
    print("Data shape：", iris.data.shape)
    print("Label names：", iris.target_names)
    print("Label shape：", iris.target.shape)

    print("Labels of 50 to 60th：", iris.target[50:60])
    print("Sample data at 0：", iris.data[0])
    print("Sample label at 0：", iris.target_names[iris.target[0]])