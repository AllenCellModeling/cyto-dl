from collections.abc import MutableMapping
from typing import Dict, Optional, Union

import numpy as np
from monai.transforms import Transform

from . import filter_columns


class GroupCols(Transform):
    def __init__(
        self,
        groups: Dict[str, Optional[Union[str, Dict]]],
        dtypes: Optional[Dict[str, Optional[Union[str, np.dtype, type]]]] = None,
    ):
        """
        Parameters
        ----------
        groups: Dict[str, Optional[Union[str, Dict]]]
            Dictionary where keys are column group names (which become batch keys)
            and values are either:
            - a dictionary containing the kwargs to be used in a col to `filter_cols`
            - a string, to use a single column in that group
            - `None`, to use a single column, with the same name as the key

        dtypes: Dict[str, Optional[Union[str, np.dtype]]],
            Dictionary where keys are column group names (as in `groups`),
            and values are either:
            - a numpy dtype or python type
            - a string recognize by `np.dtype` that can be turned into a dtype
            - `None` in which case the dtype of the group remains as-is

        """
        super().__init__()
        self.groups = groups
        self.dtypes = dtypes or {}

    def _make_group(self, k, v, row):
        if not isinstance(v, (str, MutableMapping)) and v is not None:
            raise TypeError(
                f"Values in `groups` must be either `str`, "
                f"`MutableMapping` (e.g. `dict`), or `None`. "
                f"Got {type(v)}"
            )

        if v is None:
            return row[k]
        if isinstance(v, str):
            return row[v]
        return filter_columns(row, **v)

    def __call__(self, row):
        res = {}
        for k, v in self.groups.items():
            if v is None:
                res[k] = row[k]
            elif isinstance(v, str):
                res[k] = row[v]
            elif isinstance(v, MutableMapping):
                res[k] = filter_columns(row, **v)
            else:
                raise TypeError(
                    f"Values in `groups` must be either `str`, "
                    f"`MutableMapping` (e.g. `dict`), or `None`. "
                    f"Got `{type(v)}` for key '{k}'"
                )

            _dtype = self.dtypes.get(k)
            if _dtype is not None:
                if isinstance(_dtype, str):
                    _dtype = np.dtype(_dtype)

                if isinstance(_dtype, np.dtype):
                    _dtype = _dtype.type

                if not isinstance(_dtype, type):
                    raise TypeError(
                        f"Values in `dtypes` must be either `str`, "
                        f"`np.dtype`, `type` or `None`. "
                        f"Got `{type(_dtype)}` for key '{k}'"
                    )

                res[k] = _dtype(res[k])

        return res
