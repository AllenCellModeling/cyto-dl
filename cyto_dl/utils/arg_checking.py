from hydra.utils import get_class
from numpy.typing import DTypeLike


def get_dtype(dtype: DTypeLike) -> DTypeLike:
    if isinstance(dtype, str):
        return get_class(dtype)
    elif dtype is None:
        return dtype
    elif isinstance(dtype, type):
        return dtype
    else:
        raise ValueError(f"Expected dtype to be DtypeLike, string, or None, got {type(dtype)}")
