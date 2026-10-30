from sklearn.datasets import load_iris


project_name = 'Iris ML Classifier'

features = ['sepal_length',
            'sepal_width',
            'petal_length',
            'petal_width']

def show_project_info():
    print('Project Name:', project_name)
    print('Features number:', len(features))
    for feature in features:
        print(feature)




if __name__ == '__main__':
    show_project_info()
    iris = load_iris()
    print(iris.feature_names)
    print(iris.data)
    print(iris.target_names)
    print(iris.target.shape)