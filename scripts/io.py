import numpy as np
import pandas as pd
import os


def read_metrics(path):
    '''
    Given a path, converts the metrics.dat into pd.df
    '''
    df= pd.read_csv(os.path.join(path , 'metrics.dat'))
    df = df.groupby(['#epoch'], as_index=False).last()
    return df


def read_dat(path, filename):
    """
    Read a whitespace-separated .dat file into a pandas DataFrame.
    """
    full_path = os.join.path(path, filename)

    if not full_path.exists():
        print(f"[Warning] File not found: {full_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(full_path, sep=r"\s+")
        if df.empty:
            print(f"[Warning] Empty file: {full_path}")
            return pd.DataFrame()
        return df

    except pd.errors.EmptyDataError:
        print(f"[Warning] Empty or unreadable file: {full_path}")
        return pd.DataFrame()
