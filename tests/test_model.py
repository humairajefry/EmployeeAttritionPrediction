from data.data_loader import load_data
from preprocessing.preprocessing import preprocess_data, split_data
from models.train_model import train_logistic_regression


def test_model_training():
    df = load_data()
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_logistic_regression(X_train, y_train)

    assert model is not None