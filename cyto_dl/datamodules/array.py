from typing import Callable, Dict, List, Sequence, Union

import numpy as np
from monai.data import DataLoader, Dataset
from monai.transforms import Compose
from omegaconf import ListConfig, OmegaConf


def make_array_dataloader(
    data: Union[np.ndarray, List[np.ndarray], List[Dict[str, np.ndarray]]],
    transforms: Union[Sequence[Callable], Callable],
    source_key: str = "input",
    **dataloader_kwargs,
):
    """Create a dataloader from a an array dataset.

    Parameters
    ----------
    data:  Union[np.ndarray, List[np.ndarray], List[Dict[str, np.ndarray]],
        If a numpy array (prediction only), the dataloader will be created with a single source_key.
        If a list each element must be a numpy array (for prediction) or a dictionary containing numpy array values (for training).

    transforms: Union[Sequence[Callable], Callable],
        Transforms to apply to each sample

    dataloader_kwargs:
        Additional keyword arguments are passed to the
        torch.utils.data.DataLoader class when instantiating it (aside from
        `shuffle` which is only used for the train dataloader).
        Among these args are `num_workers`, `batch_size`, `shuffle`, etc.
        See the PyTorch docs for more info on these args:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    if isinstance(transforms, (list, tuple, ListConfig)):
        transforms = Compose(transforms)
    data = OmegaConf.to_object(data)
    if isinstance(data, (list, tuple, ListConfig)):
        data = [{source_key: d} if isinstance(d, np.ndarray) else d for d in data]
    elif isinstance(data, np.ndarray):
        data = [{source_key: data}]
    else:
        raise ValueError(
            f"Invalid data type: {type(data)}. Data must be a numpy array or list of numpy arrays."
        )

    dataset = Dataset(data, transform=transforms)

    return DataLoader(dataset, **dataloader_kwargs)
