import aicssegmentation
import numpy as np
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from scipy.stats import norm
import torch
from typing import List, Dict
from monai.data.meta_tensor import MetaTensor



class MeanNormalizeIntensity(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: List[str], lower_bound: float, upper_bound: float) -> None:
        super().__init__()
        self.keys = keys
        self.lbound = lower_bound
        self.ubound = upper_bound
        
    def normalize(self, img: NdarrayOrTensor):
        
        is_tensor = not isinstance(img, np.ndarray)
        is_metatensor = not isinstance(img, MetaTensor)
        if is_tensor:
            im = np.array(img)
        elif is_metatensor:
            im = np.array(img.get_array(out_type=np.ndarray))
            
        m, s = norm.fit(im.flat)
        strech_min = max(m - self.lbound * s, im.min())
        strech_max = min(m + self.ubound * s, im.max())
        im[im > strech_max] = strech_max
        im[im < strech_min] = strech_min
        im = (im - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
        
        if is_tensor:
            img = torch.from_numpy(im)
        elif is_metatensor:
            img.set_array(torch.from_numpy(im))
        return img
        
    def __call__(self, img_dict: Dict[str, NdarrayOrTensor]) -> NdarrayOrTensor:
        
        for key in img_dict.keys():
            if key in self.keys:
                img_dict[key] = self.normalize(img_dict[key])
        
        return img_dict