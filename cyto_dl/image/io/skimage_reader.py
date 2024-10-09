"""ADAPTED FROM https://github.com/MMV-Lab/mmv_im2im/"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from monai.config import PathLike
from monai.data import ImageReader
from monai.data.image_reader import _stack_images
from monai.utils import ensure_tuple, require_pkg
from skimage.io import imread


@require_pkg(pkg_name="skimage")
class SkimageReader(ImageReader):
    def __init__(
        self,
        channels: Optional[list] = None,
        transforms: Optional[list] = None,
    ):
        super().__init__()
        self.channels = channels
        self.transforms = transforms

    def read(self, data: Union[Sequence[PathLike], PathLike]):
        filenames: Sequence[PathLike] = ensure_tuple(data)
        img_ = []
        for name in filenames:
            this_im = imread(name)
            if self.channels:
                this_im = this_im[self.channels]

            if self.transforms:
                for transform in self.transforms:
                    this_im = transform(this_im)
            img_.append(this_im)

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        img_array: List[np.ndarray] = []
        for img_obj in ensure_tuple(img):
            data = img_obj
            img_array.append(data)

        return _stack_images(img_array, {}), {}

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        return True
