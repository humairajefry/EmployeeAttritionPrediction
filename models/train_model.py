from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(
    X_train,
    y_train
):
    """
    Train a Logistic Regression model.

    Parameters:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        LogisticRegression: Trained model.
    """

    model = LogisticRegression()

    model.fit(X_train, y_train)

    return model


def train_random_forest(
    X_train,
    y_train
):
    """
    Train a Random Forest classifier.

    Parameters:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    
    model = RandomForestClassifier(
        random_state=75
    )

    model.fit(X_train, y_train)

    return model