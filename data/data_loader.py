import pandas as pd

def load_data():
    """
    Load the employee attrition dataset from CSV.

    Returns:
        pandas.DataFrame: Employee attrition dataset.
    """

    df = pd.read_csv(
        "data/employee_attrition.csv"
    )

    return df