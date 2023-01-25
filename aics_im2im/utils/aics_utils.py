import aicssegmentation
import numpy as np
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from scipy.stats import norm
import torch
from typing import List, Dict



class MeanNormalizeIntensity(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: List[str], lower_bound: float, upper_bound: float) -> None:
        super().__init__()
        self.keys = keys
        self.lbound = lower_bound
        self.ubound = upper_bound
        
    def normalize(self, img):
        is_tensor = not isinstance(img, np.ndarray)
        if is_tensor:
            img = np.array()
            
        m, s = norm(img.flat)
        strech_min = max(m - self.lbound * s, img.min())
        strech_max = min(m + self.ubound * s, img.max())
        img[img > strech_max] = strech_max
        img[img < strech_min] = strech_min
        img = (img - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
        
        if is_tensor:
            img = torch.from_numpy(img)
        
    def __call__(self, img_dict: Dict[str, NdarrayOrTensor]) -> NdarrayOrTensor:
        
        for key in img_dict.keys():
            if key in self.keys:
                img_dict[key] = self.normalize(img_dict[key])
        
        return img_dict