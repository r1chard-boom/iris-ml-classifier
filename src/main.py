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
    pass

if __name__ == '__main__':
    show_project_info()
