from typing import Optional

import pandas as pd


def split_dataframe(
    dataframe: pd.DataFrame,
    train_frac: float,
    val_frac: Optional[float] = None,
    return_splits: bool = True,
    seed: int = 42,
):
    """Given a pandas dataframe, perform a train-val-test split and either return three different
    dataframes, or append a column identifying the split each row belongs to.

    TODO: extend this to enable balanced / stratified splitting

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe

    train_frac: float
        Fraction of data to use for training. Must be <= 1

    val_frac: Optional[float]
        Fraction of data to use for validation. By default,
        the data not used for training is split in half
        between validation and test

    return_splits: bool = True
        Whether to return the three splits separately, or to append
        a column to the existing dataframe and return the modified
        dataframe

    seed: int = 42
        Random seed for reproducibility
    """

    # import here to optimize CLIs / Fire usage
    from sklearn.model_selection import train_test_split

    train_ix, val_test_ix = train_test_split(
        dataframe.index.tolist(), train_size=train_frac, random_state=seed
    )
    if val_frac is not None:
        val_frac = val_frac / (1 - train_frac)
    else:
        # by default use same size for val and test
        val_frac = 0.5

    val_ix, test_ix = train_test_split(val_test_ix, train_size=val_frac, random_state=seed)

    if return_splits:
        return dict(
            train=dataframe.loc[train_ix],
            valid=dataframe.loc[val_ix],
            test=dataframe.loc[test_ix],
        )

    dataframe.loc[train_ix, "split"] = "train"
    dataframe.loc[val_ix, "split"] = "valid"
    dataframe.loc[test_ix, "split"] = "test"

    return dataframe


def sample_n_each(
    dataframe: pd.DataFrame,
    column: str,
    number: int = 1,
    force: bool = False,
    seed: int = 42,
):
    """Transform a dataframe to have equal number of rows per value of `column`.

    In case a given value of `column` has less than `number` corresponding rows:
    - if `force` is True the corresponding rows are sampled with replacement
    - if `force` is False all the rows are given for that value

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe

    column: str
        The column to be used for selection

    number: int
        Number of rows to include per unique value of `column`

    force: bool = False
        Toggle upsampling of classes with number of samples smaller
        than `number`

    seed: int
        Random seed used for sampling
    """

    values = dataframe[column].unique()

    subsets = []
    for value in values:
        class_rows = dataframe[dataframe[column] == value]
        if force or (len(class_rows) >= number):
            subsets.append(
                class_rows.sample(
                    number,
                    random_state=seed,
                    # only sample with replacement if there
                    # aren't enough data points in this class
                    replace=(len(class_rows) < number),
                )
            )
        else:
            subsets.append(class_rows.sample(frac=1, random_state=seed))

    return pd.concat(subsets)
