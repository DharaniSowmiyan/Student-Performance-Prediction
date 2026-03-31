import pandas as pd


def load_clean_logs():
    return pd.read_csv("../data/processed/clean_logs.csv")