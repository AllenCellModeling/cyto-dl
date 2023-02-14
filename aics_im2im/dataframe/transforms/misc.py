from typing import Sequence

import numpy as np
import pandas as pd


def append_one_hot(dataframe: pd.DataFrame, column: str):
    """Modifies its argument by appending the one hot encoding columns into the given dataframe.
    Calls function one_hot_encoding.

    Parameters
    -----------
    dataframe: pd.DataFrame
        Input dataframe

    column: str
        Column to convert into one hot encoding
    """

    # import here to optimize CLIs / Fire usage
    from sklearn.preprocessing import OneHotEncoder

    one_hot = OneHotEncoder(sparse=False).fit_transform(dataframe[[column]])

    for idx in range(one_hot.shape[1]):
        dataframe[f"{column}_one_hot_{idx}"] = one_hot[:, idx]

    return dataframe


def append_labels_to_integers(dataframe: pd.DataFrame, column: str):
    """Modifies its argument by appending the integer-encoded values of `column` into the given
    dataframe.

    Parameters
    -----------
    dataframe: pd.DataFrame
        Input dataframe

    column: str
        Column to convert into one hot encoding
    """

    # import here to optimize CLIs / Fire usage
    from sklearn.preprocessing import LabelEncoder

    dataframe[f"{column}_integer"] = LabelEncoder().fit_transform(dataframe[[column]])

    return dataframe


def append_class_weights(dataframe: pd.DataFrame, column: str):
    """Add class weights (based on `column`) to a dataframe.

    Parameters
    -----------
    dataframe: pd.DataFrame
        Input dataframe

    column: str
        Column to base the weights on
    """
    labels_unique, counts = np.unique(dataframe[column], return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    class_weights_dict = dict(zip(labels_unique, class_weights))
    weights = [class_weights_dict[e] for e in dataframe[column]]
    dataframe[f"{column}_class_weights"] = weights
    return dataframe


def make_random_df(columns: Sequence[str] = list("ABCD"), n_rows: int = 100):
    """Generate a random dataframe. Useful to test data wrangling pipelines.

    Parameters
    ----------
    columns: Sequence[str] = ["A","B","C","D"]
        List of columns to add to the random dataframe. If none are provided,
        a dataframe with columns ["A","B","C","D"] is created

    n_rows: int = 100
        Number of rows to create for the random dataframe
    """

    data = np.random.randn(n_rows, len(columns))
    return pd.DataFrame(data, columns=columns)
