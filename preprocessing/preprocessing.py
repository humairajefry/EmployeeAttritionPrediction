import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split


def preprocess_data(df):

    # Drop unnecessary columns
    df = df.drop(
        ['StandardHours', 'EmployeeCount'],
        axis=1
    )

    df = df.drop('Over18', axis=1)

    # Encode target column
    target_encoder = LabelEncoder()

    df['Attrition'] = target_encoder.fit_transform(
        df['Attrition']
    )

    # One-hot encode categorical columns
    categorical_columns = [
        'BusinessTravel',
        'Department',
        'EducationField',
        'Gender',
        'JobRole',
        'MaritalStatus',
        'OverTime'
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    return df


def split_data(df):

    X = df.drop('Attrition', axis=1)

    y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=75
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test