import pandas as pd

def load_data():
    """
    Load the employee attrition dataset from CSV.

    Returns:
        pandas.DataFrame: Employee attrition dataset.
    """

    df = pd.read_csv(
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    return df