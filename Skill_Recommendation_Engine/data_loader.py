import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)

    # remove null values
    data = data.dropna()

    return data