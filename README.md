# Employee Attrition Prediction

## Project Overview

This project predicts employee attrition using machine learning techniques based on the IBM HR Analytics Employee Attrition dataset.

The original project was obtained from a publicly available Kaggle notebook and was restructured into a modular Python project as part of the SF35803 Computer Programming Individual Coursework. The objective was to improve code organization, readability, maintainability, testing, and documentation by separating the machine learning workflow into independent modules.

---

## Dataset

Dataset: IBM HR Analytics Employee Attrition Dataset

The dataset contains employee-related information such as demographic details, job roles, work experience, compensation, and job satisfaction indicators. The target variable is **Attrition**, which indicates whether an employee has left the company.

---

## Project Objectives

* Load and process employee data.
* Perform data preprocessing and feature encoding.
* Train machine learning models for attrition prediction.
* Evaluate model performance using classification metrics.
* Visualize model results using confusion matrices.
* Demonstrate modular software development practices.

---

## Project Structure

employee_attrition_prediction/

├── data/
│   ├── **init**.py
│   └── data_loader.py
│   └── employee_attrition.csv
│
├── preprocessing/
│   ├── **init**.py
│   └── preprocessing.py
│
├── models/
│   ├── **init**.py
│   └── train_model.py
│
├── evaluation/
│   ├── **init**.py
│   └── evaluation.py
│
├── utils/
│   ├── **init**.py
│   └── visualization.py
│
├── tests/
│   ├── **init**.py
│   └── test_model.py
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt

---

## Machine Learning Workflow

### 1. Data Loading

The dataset is loaded from a CSV file using the data loading module.

### 2. Data Preprocessing

The preprocessing stage includes:

* Removing unnecessary features
* Label encoding the target variable
* One-hot encoding categorical features
* Feature scaling
* Train-test splitting

### 3. Model Training

The following machine learning models are implemented:

* Logistic Regression
* Random Forest Classifier

### 4. Model Evaluation

Model performance is evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix

### 5. Visualization

Confusion matrix heatmaps are generated to visually assess classification performance.

---

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the main application:

```bash
python main.py
```

---

## Running Unit Tests

Run the test suite using:

```bash
pytest
```

or

```bash
PYTHONPATH=. pytest
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Pytest

---

## Author

Nur Humaira

SF35803 Computer Programming

Industrial Physics Programme

Universiti Malaysia Sabah
