"""ADAPTED FROM https://github.com/MMV-Lab/mmv_im2im/"""

import logging

logging.getLogger("ome_zarr").setLevel(logging.WARNING)
logging.getLogger("ome_zarr.reader").setLevel(logging.WARNING)
logging.getLogger("bfio.init").setLevel(logging.ERROR)
logging.getLogger("bfio.backends").setLevel(logging.ERROR)
logging.getLogger("xmlschema").setLevel(logging.ERROR)

import warnings

warnings.filterwarnings("ignore")

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from aicspylibczi import CziFile
from monai.config import PathLike
from monai.data import ImageReader
from monai.data.image_reader import _stack_images
from monai.utils import ensure_tuple, require_pkg


@require_pkg(pkg_name="aicspylibczi")
class CziReader(ImageReader):
    def __init__(self, **reader_kwargs):
        super().__init__()
        self.reader_kwargs = reader_kwargs

    def read(self, data: Union[Sequence[PathLike], PathLike]):
        filenames: Sequence[PathLike] = ensure_tuple(data)
        img_ = []
        for name in filenames:
            img_.append(CziFile(f"{name}"))

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        img_array: List[np.ndarray] = []

        for img_obj in ensure_tuple(img):
            data = img_obj.read_image(**self.reader_kwargs)[0].squeeze()
            img_array.append(data)

        return _stack_images(img_array, {}), {}

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        return True
