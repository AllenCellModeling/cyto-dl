from typing import Callable, Optional, Sequence, Union

from monai.data import DataLoader, Dataset, PersistentDataset
from monai.transforms import Compose
from omegaconf import ListConfig
from upath import UPath as Path

from cyto_dl.dataframe.transforms.filter import _filter_columns as filter_filenames


def make_folder_dataloader(
    path: Union[Path, str],
    transforms: Union[Sequence[Callable], Callable],
    cache_dir: Optional[Union[Path, str]] = None,
    regex: Optional[str] = None,
    startswith: Optional[str] = None,
    endswith: Optional[str] = None,
    contains: Optional[str] = None,
    excludes: Optional[str] = None,
    **dataloader_kwargs,
):
    """Create a dataloader based on a folder of samples. If no transforms are applied, each sample
    is a dictionary with a key "input" containing the corresponding path and a key "orig_fname"
    containing the original filename (with no extension).

    Files can be filtered out of the list with name-based rules, using `regex`,
    `startswith`, `endswith`, `contains`, `excludes`.

    Parameters
    ----------
    path: Union[Path, str],
        Path to folder

    transforms: Union[Sequence[Callable], Callable],
        Transforms to apply to each sample

    cache_dir: Optional[Union[Path, str]] = None
        Path to a directory in which to store cached transformed inputs, to
        accelerate batch loading.

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

    data = filter_filenames(
        list(map(str, Path(path).glob("*"))),
        regex,
        startswith,
        endswith,
        contains,
        excludes,
    )

    data = [{"input": path, "orig_fname": Path(path).stem} for path in data]

    if cache_dir is not None:
        dataset = PersistentDataset(data, transform=transforms, cache_dir=cache_dir)
    else:
        dataset = Dataset(data, transform=transforms)

    return DataLoader(dataset, **dataloader_kwargs)
