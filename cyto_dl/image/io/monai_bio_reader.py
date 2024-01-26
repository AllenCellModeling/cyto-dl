"""ADAPTED FROM https://github.com/MMV-Lab/mmv_im2im/"""

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from aicsimageio import AICSImage
from monai.config import PathLike
from monai.data import ImageReader
from monai.data.image_reader import _stack_images
from monai.utils import ensure_tuple, require_pkg
from omegaconf import Container, OmegaConf


@require_pkg(pkg_name="aicsimageio")
class MonaiBioReader(ImageReader):
    def __init__(self, **reader_kwargs):
        super().__init__()
        self.reader_kwargs = {
            k: OmegaConf.to_container(v) if isinstance(v, Container) else v
            for k, v in reader_kwargs.items()
            if v is not None
        }

    def read(self, data: Union[Sequence[PathLike], PathLike]):
        filenames: Sequence[PathLike] = ensure_tuple(data)
        img_ = []
        for name in filenames:
            img_.append(AICSImage(f"{name}"))

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        img_array: List[np.ndarray] = []

        for img_obj in ensure_tuple(img):
            data = img_obj.get_image_dask_data(**self.reader_kwargs).compute()
            img_array.append(data)

        return _stack_images(img_array, {}), {}

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        return True
