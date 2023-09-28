from typing import Callable, List, Optional, Sequence, Union

from monai.data import DataLoader, Dataset, PersistentDataset
from monai.transforms import Compose
from omegaconf import DictConfig, ListConfig, OmegaConf
from upath import UPath as Path


def make_data_dict_dataloader(
    data: Sequence[Union[DictConfig, dict]],
    transforms: Union[Sequence[Callable], Callable],
    cache_dir: Optional[Union[Path, str]] = None,
    **dataloader_kwargs,
):
    """Create a dataloader based on a dictionary of paths to images.

    Parameters
    ----------
    data: Sequence[Union[DictConfig, dict]]
        A sequence of dictionaries, each containing a key expected by transforms
        (usually an image path)

    transforms: Union[Sequence[Callable], Callable],
        Transforms to apply to each sample

    cache_dir: Optional[Union[Path, str]] = None
        Path to a directory in which to store cached transformed inputs, to
        accelerate batch loading.

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

    if isinstance(data, (ListConfig, DictConfig)):
        data = OmegaConf.to_container(data)

    if cache_dir is not None:
        dataset = PersistentDataset(data, transform=transforms, cache_dir=cache_dir)
    else:
        dataset = Dataset(data, transform=transforms)
    return DataLoader(dataset, **dataloader_kwargs)
