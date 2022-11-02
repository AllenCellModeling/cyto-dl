#### TAKEN FROM https://github.com/MMV-Lab/mmv_im2im/blob/main/LICENSE


from typing import Union, Dict, List
import numpy as np
from aicsimageio import AICSImage
from typing import Sequence, Tuple
from monai.data import ImageReader
from monai.utils import ensure_tuple, require_pkg
from monai.config import PathLike
from monai.data.image_reader import _stack_images
import warnings

warnings.filterwarnings("once")


@require_pkg(pkg_name="aicsimageio")
class monai_bio_reader(ImageReader):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def read(self, data: Union[Sequence[PathLike], PathLike]):
        filenames: Sequence[PathLike] = ensure_tuple(data)
        img_ = []
        for name in filenames:
            img_.append(AICSImage(f"{name}"))

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        img_array: List[np.ndarray] = []

        for img_obj in ensure_tuple(img):
            data = img_obj.get_image_dask_data("ZYX", C=0, T=0).compute()
            img_array.append(data)

        return _stack_images(img_array, {}), {}

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        return True