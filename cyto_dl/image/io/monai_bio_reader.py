"""ADAPTED FROM https://github.com/MMV-Lab/mmv_im2im/"""

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from bioio import BioImage
from monai.config import PathLike
from monai.data import ImageReader
from monai.data.image_reader import _stack_images
from monai.utils import ensure_tuple, require_pkg
from omegaconf import Container, OmegaConf


@require_pkg(pkg_name="bioio")
class MonaiBioReader(ImageReader):
    def __init__(self, dask_load: bool = True, **reader_kwargs):
        """
        dask_load: bool = True
            Whether to use dask to load images. If False, full images are loaded into memory before extracting specified scenes/timepoints.
        reader_kwargs: Dict
            Additional keyword arguments to pass to BioImage.get_image_data or BioImage.get_image_dask_data
        """
        super().__init__()
        self.reader_kwargs = {
            k: OmegaConf.to_container(v) if isinstance(v, Container) else v
            for k, v in reader_kwargs.items()
            if v is not None
        }
        self.dask_load = dask_load

    def read(self, data: Union[Sequence[PathLike], PathLike]):
        filenames: Sequence[PathLike] = ensure_tuple(data)
        img_ = [BioImage(name) for name in filenames]
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        img_array: List[np.ndarray] = []

        for img_obj in ensure_tuple(img):
            if self.dask_load:
                data = img_obj.get_image_dask_data(**self.reader_kwargs).compute()
            else:
                data = img_obj.get_image_data(**self.reader_kwargs)
            img_array.append(data)

        return _stack_images(img_array, {}), {}

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        return True
