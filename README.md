# Iris ML Classifier

A machine learning classification project based on the Iris dataset using Python and Scikit-learn.

## 1. Project Overview

This project implements a complete machine learning workflow for classifying Iris flowers into three categories:

- Setosa
- Versicolor
- Virginica

The project covers data exploration, model training, feature standardization, model evaluation, model persistence, and inference.

## 2. Technologies

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib
- Git / GitHub

## 3. Dataset

The Iris dataset contains 150 samples, with four features per sample:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

The dataset is divided into training and testing sets using an 80/20 split, with `random_state=42` and stratified sampling.

## 4. Project Structure

```text
iris-ml-classifier/
├── docs/
│   └── experiments.md
├── models/
├── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── visualize.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/r1chard-boom/iris-ml-classifier.git
cd iris-ml-classifier
```

Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 6. Usage

### Train and Evaluate

Run the training script:

```bash
python -m src.train
```

The script trains a Pipeline containing StandardScaler and LogisticRegression, evaluates the model, and saves the trained Pipeline to:

```text
models/iris_pipeline.joblib
```

### Predict New Samples

After training, run:

```bash
python -m src.predict
```

The script loads the saved Pipeline and predicts the class of a new Iris sample.

## 7. Experimental Results

Two classification approaches were compared:

| Model | Accuracy |
|---|---|
| Logistic Regression | 96.67% |
| StandardScaler + Logistic Regression | 93.33% |

The final Pipeline obtained an accuracy of 93.33% on the 30-sample test set.

The result illustrates that feature standardization does not necessarily improve accuracy on every dataset and data split.

Detailed experimental configurations and analysis are available in `docs/experiments.md`.

## 8. What I Learned

Through this project, I practiced:

- Loading and exploring datasets with Pandas
- Splitting data into training and testing sets
- Training classification models with Scikit-learn
- Evaluating models using Accuracy, Precision, Recall, F1-score, and confusion matrices
- Using StandardScaler and Pipeline
- Saving and loading trained models with Joblib
- Organizing Python code into reusable modules
- Recording and interpreting machine learning experiments
- Managing project versions using Git and GitHub

## 9. Future Improvements

Possible next steps include:

- Adding cross-validation
- Performing hyperparameter tuning
- Comparing additional classification algorithms
- Adding automated tests
- Applying the same workflow to industrial fault diagnosis datasets

## 10. Author

A machine learning practice project developed as part of my AI engineering learning roadmap.****