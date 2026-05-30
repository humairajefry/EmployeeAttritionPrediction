from data.data_loader import load_data

from preprocessing.preprocessing import (
    preprocess_data,
    split_data
)

from models.train_model import (
    train_logistic_regression,
    train_random_forest
)

from evaluation.evaluation import (
    evaluate_model
)

from utils.visualization import (
    plot_confusion_matrix
)


def main():

    # Load dataset
    df = load_data()

    # Preprocess dataset
    df = preprocess_data(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(df)

    # Logistic Regression
    logistic_model = train_logistic_regression(
        X_train,
        y_train
    )

    print("\nLOGISTIC REGRESSION RESULTS")

    evaluate_model(
        logistic_model,
        X_test,
        y_test
    )

    plot_confusion_matrix(
        logistic_model,
        X_test,
        y_test,
        "logistic_confusion_matrix.png"
    )

    # Random Forest
    rf_model = train_random_forest(
        X_train,
        y_train
    )

    print("\nRANDOM FOREST RESULTS")

    evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    plot_confusion_matrix(
        rf_model,
        X_test,
        y_test,
        "random_forest_confusion_matrix.png"
    )


if __name__ == "__main__":
    main()