"""ADAPTED FROM https://github.com/MMV-Lab/mmv_im2im/"""

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from aicsimageio import AICSImage
from monai.config import PathLike
from monai.data import ImageReader
from monai.data.image_reader import _stack_images
from monai.utils import ensure_tuple, require_pkg


@require_pkg(pkg_name="aicsimageio")
class VarianceReader(ImageReader):
    """
    Reader for images from Variance dataset
    """
    def __init__(self, channels = ['Cell', 'Nuc', 'Struct']):
        """
        Parameters
        ----------
        channels: List[str]
            List of channels to read from Variance dataset. Options are ['Cell', 'Nuc', 'Struct', 'Brightfield']
        """
        super().__init__()

        name_map = {'Cell': ['cmdrp'], 'Nuc':['h3342'], 'Struct': ['egfp', 'mtagrfpt'], 'Brightfield':['bright100', 'bright100x', 'tl100x', 'bright2']}
        self.channels = []
        for ch in channels: 
            assert ch in ['Cell', 'Nuc', 'Struct', 'Brightfield']
            self.channels.append(name_map[ch])


    def read(self, data: Union[Sequence[PathLike], PathLike]):
        filenames: Sequence[PathLike] = ensure_tuple(data)
        img_ = []
        for name in filenames:
            img_.append(AICSImage(f"{name}"))

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        img_array: List[np.ndarray] = []
        # remove spaces and underscores
        img_channel_names = [cn.lower().replace(' ', '').replace('_', '') for cn in img.channel_names] 
        channel_indices = []
        for ch_type in self.channels:
            for alias in ch_type:
                if alias in img_channel_names:
                    channel_indices.append(img_channel_names.index(alias))
                    break
            else:
                raise ValueError(f"None of {ch_type} found in image, available channels are {img_channel_names}")
            
        for img_obj in ensure_tuple(img):
            data = img_obj.get_image_dask_data('CZYX', C=channel_indices).compute()
            img_array.append(data)
        return _stack_images(img_array, {}), {}

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        return True
