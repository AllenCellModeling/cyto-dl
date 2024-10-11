import re
from contextlib import suppress
from pathlib import Path
from typing import Optional, Sequence, Union

from upath import UPath

try:
    import modin.pandas as pd
except ModuleNotFoundError:
    import pandas as pd

import anndata
import pyarrow.parquet


def read_h5ad(path, include_columns=None, backed=None):
    """Read an annData object stored in a .h5ad file.

    Parameters
    ----------

    path: Union[Path, str]
        Path to the .h5ad file

    include_columns: Optional[Sequence[str]] = None
        List of column names and/or regex expressions, used to only include the
        desired columns in the resulting dataframe.

    backed: Optional[str] = None
        Can be (either "r" or "r+").
        See anndata's docs for details:
        https://anndata.readthedocs.io/en/latest/generated/anndata.read_h5ad.html#anndata.read_h5ad

    Returns
    -------
    annData
    """

    if backed:
        assert backed in ("r", "r+")
    dataframe = anndata.read_hda5(path, backed=backed)

    if include_columns is not None:
        columns = []
        for filter_ in include_columns:
            columns += filter_columns(dataframe.obs.columns.tolist(), regex=filter_)

        dataframe.obs = dataframe.obs[columns]
    return dataframe


def read_parquet(path, include_columns=None):
    """Read a dataframe stored in a .parquet file, and optionally include only the columns given by
    `include_columns`

    Parameters
    ----------

    path: Union[Path, UPath, str]
        Path to the .parquet file
    include_columns: Optional[Sequence[str]] = None
        List of column names and/or regex expressions, used to only include the
        desired columns in the resulting dataframe.

    Returns
    -------

    dataframe: pd.DataFrame
    """
    if include_columns is not None:
        schema = pyarrow.parquet.read_schema(path, memory_map=True)

        columns = []
        for filter_ in include_columns:
            columns += filter_columns(schema.names, regex=filter_)
    else:
        columns = None

    return pd.read_parquet(path, columns=columns)


def read_csv(path, include_columns=None):
    """Read a dataframe stored in a .csv file, and optionally include only the columns given by
    `include_columns`

    Parameters
    ----------

    path: Union[Path, UPath, str]
        Path to the .csv file
    include_columns: Optional[Sequence[str]] = None
        List of column names and/or regex expressions, used to only include the
        desired columns in the resulting dataframe.

    Returns
    -------

    dataframe: pd.DataFrame
    """

    dataframe = pd.read_csv(path)

    if include_columns is not None:
        columns = []
        for filter_ in include_columns:
            columns += filter_columns(dataframe.columns.tolist(), regex=filter_)

        dataframe = dataframe[columns]

    return dataframe


def read_dataframe(
    dataframe: Union[Path, UPath, str, pd.DataFrame],
    required_columns: Optional[Sequence[str]] = None,
    include_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load a dataframe from a .csv or .parquet file, or assert a given pd.DataFrame contains the
    expected required columns.

    Parameters
    ----------

    dataframe: Union[Path, UPath, str, pd.DataFrame]
        Either the path to the dataframe to be loaded, or a pd.DataFrame. Supported
        file types are .csv and .parquet

    required_columns: Optional[Sequence[str]] = None
        List of columns that the dataframe must contain. If these aren't found,
        a ValueError is thrown

    include_columns: Optional[Sequence[str]] = None
        List of column names and/or regex expressions, used to only include the
        desired columns in the resulting dataframe. If `required_columns` is not
        None, those get appended to `include_columns` (without duplication).

    Returns
    -------

    dataframe: pd.DataFrame
    """

    required_columns = set(required_columns) if required_columns else set()
    include_columns = set(include_columns) if include_columns else set()

    include_columns = include_columns | required_columns

    include_columns = sorted(list(include_columns))
    required_columns = sorted(list(required_columns))

    if not include_columns:
        include_columns = None

    if isinstance(dataframe, str):
        dataframe = UPath(dataframe)

    if isinstance(dataframe, (UPath, Path)):
        with suppress((NotImplementedError, FileNotFoundError)):
            dataframe = dataframe.expanduser().resolve(strict=True)

        if not dataframe.is_file():
            raise FileNotFoundError("Manifest file not found at given path")

        if dataframe.suffix == ".csv":
            dataframe = read_csv(dataframe, include_columns)
        elif dataframe.suffix == ".parquet":
            dataframe = read_parquet(dataframe, include_columns)
        elif dataframe.suffix == ".h5ad":
            dataframe = read_h5ad(dataframe, include_columns)
        else:
            raise TypeError("File type of provided manifest is not in [.csv, .parquet, .h5ad]")

    elif isinstance(dataframe, pd.DataFrame):
        if include_columns is not None:
            columns = []
            for filter_ in include_columns:
                columns += filter_columns(dataframe.columns, regex=filter_)

            dataframe = dataframe[columns]
    elif isinstance(dataframe, anndata.AnnData):
        if include_columns is not None:
            columns = []
            for filter_ in include_columns:
                columns += filter_columns(dataframe.obs.columns, regex=filter_)

            dataframe.obs = dataframe.obs[columns]
    else:
        raise TypeError(
            f"`dataframe` must be either a pd.DataFrame or a path to "
            f"a file to load one. You passed {type(dataframe)}"
        )

    if isinstance(dataframe, anndata.AnnData):
        # Make dataframe out of anndata object
        X = dataframe.X.toarray()
        X = pd.DataFrame(X, columns=[f"X_{i}" for i in range(X.shape[1])]).reset_index(drop=True)
        index = pd.DataFrame(dataframe.obs_names)
        dataframe = dataframe.obs.reset_index(drop=True)
        dataframe = pd.concat([X, dataframe], axis=1)
        dataframe = pd.concat([index, dataframe], axis=1)

    if required_columns is not None:
        missing_columns = set(required_columns) - set(dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"Some or all of the required columns were not "
                f"found on the given dataframe:\n{missing_columns}"
            )

    return dataframe


def filter_columns(
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
