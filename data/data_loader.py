import pandas as pd

def load_data():

    df = pd.read_csv(
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    return df