import re
from collections.abc import MutableMapping
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd


def filter_rows(dataframe: pd.DataFrame, column: str, values: Sequence, exclude: bool = False):
    """Filter a dataframe, keeping only the rows where a given column's value is contained in a
    list of values.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe

    column: str
        The column to be used for filtering

    values: Sequence
        List of values to filter for
    """

    if exclude:
        return dataframe.loc[~dataframe[column].isin(values)]
    return dataframe.loc[dataframe[column].isin(values)]


def _filter_columns(
    columns_to_filter: Sequence[str],
    regex: Optional[str] = None,
    startswith: Optional[str] = None,
    endswith: Optional[str] = None,
    contains: Optional[str] = None,
    excludes: Optional[str] = None,
) -> Sequence[str]:
    """Filter a list of columns, using a combination of different queries, or a `regex` pattern. If
    `regex` is supplied it takes precedence and the remaining arguments are ignored. Otherwise, the
    logical AND of the supplied filters is applied, i.e. the columns that respect all of the
    supplied conditions are returned.

    Parameters
    ----------
    columns_to_filter: Sequence[str]
        List of columns to filter

    regex: Optional[str] = None
        A string containing a regular expression to be matched

    startswith: Optional[str] = None
        A substring the matching columns must start with

    endswith: Optional[str] = None
        A substring the matching columns must end with

    contains: Optional[str] = None
        A substring the matching columns must contain

    excludes: Optional[str] = None
        A substring the matching columns must not contain
    """
    if regex is not None:
        return [col for col in columns_to_filter if re.match(regex, col)]

    keep = [True] * len(columns_to_filter)
    for i in range(len(columns_to_filter)):
        if startswith is not None:
            keep[i] &= str(columns_to_filter[i]).startswith(startswith)
        if endswith is not None:
            keep[i] &= str(columns_to_filter[i]).endswith(endswith)
        if contains is not None:
            keep[i] &= contains in str(columns_to_filter[i])
        if excludes is not None:
            keep[i] &= excludes not in str(columns_to_filter[i])

    return [col for col, keep_column in zip(columns_to_filter, keep) if keep_column]


def filter_columns(
    input: Union[pd.DataFrame, Dict, Sequence[str]],
    columns: Optional[Sequence[str]] = None,
    startswith: Optional[str] = None,
    endswith: Optional[str] = None,
    contains: Optional[str] = None,
    excludes: Optional[str] = None,
    regex: Optional[str] = None,
):
    """Select columns in a dataset, using different filtering options. See
    cyto_dl.dataframe.transforms.filter_columns for more details.

    Parameters
    ----------
    input: Union[pd.DataFrame, Sequence[str]]
        The input to operate on. It can either be a pandas DataFrame,
        in which case the result is a DataFrame with only the columns
        that match the filters; or it can be a list of strings, and
        in that case the result is a list containing only the strings
        that match the filters

    columns: Optional[Sequence[str]] = None
        Explicit list of columns to include. If it is supplied,
        the remaining filters are ignored

    startswith: Optional[str] = None
        A substring the matching columns must start with

    endswith: Optional[str] = None
        A substring the matching columns must end with

    contains: Optional[str] = None
        A substring the matching columns must contain

    excludes: Optional[str] = None
        A substring the matching columns must not contain

    regex: Optional[str] = None
        A string containing a regular expression to be matched
    """
    if columns is None:
        if isinstance(input, pd.DataFrame):
            columns = input.columns.tolist()
        elif isinstance(input, MutableMapping):
            columns = [*input]
        else:
            columns = input
    columns = _filter_columns(columns, regex, startswith, endswith, contains, excludes)

    if isinstance(input, pd.DataFrame):
        return input[columns]
    elif isinstance(input, MutableMapping):
        return np.array(tuple(input[col] for col in columns))

    return columns
