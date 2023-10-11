import numpy as np
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from scipy.stats import norm
import torch
from typing import List, Dict


class AICSNormalize(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self, 
            keys: List[str], 
            mode: str,
            **kwargs
        ) -> None:
        super().__init__()
        self.keys = keys
        self.mode = mode
        if self.mode == 'mean_norm':
            assert 'lower_bound' in kwargs.keys(), print('parameter \'lower_bound\' missing from arguments for mode \'mean_norm\'')
            assert 'upper_bound' in kwargs.keys(), print('parameter \'upper_bound\' missing from arguments for mode \'mean_norm\'')
            
            self.lbound = kwargs['lower_bound']
            self.ubound = kwargs['upper_bound']
            return
        elif self.mode == 'norm_around_center':
            if 'z_center' in kwargs.keys():
                self.z_center = kwargs['z_center']
            else:
                self.z_center = None
            return
        else:
            NotImplementedError()
        
    def normalize(self, im: NdarrayOrTensor):
        is_tensor = not isinstance(im, np.ndarray)
        if is_tensor:
            im = np.array(im)
        
        if self.mode == 'mean_norm':
            im = self.mean_norm(im)
        elif self.mode == 'norm_around_center':
            im = self.norm_around_center(im)
            
        if is_tensor:
            im = torch.from_numpy(im)
        return im
        
    def norm_around_center(self, im: NdarrayOrTensor):
        if im.shape[1] < 32:
            raise ValueError("Input array must be at least length 32 in first dimension")
        if self.z_center is None:
            z_center = im.shape[1] // 2
        else:
            z_center = self.z_center
        chunk_zlen = 32
        z_start = z_center - chunk_zlen // 2
        if z_start < 0:
            z_start = 0
        if (z_start + chunk_zlen) > im.shape[1]:
            z_start = im.shape[1] - chunk_zlen
            
        if np.isinf(np.max(im)) or np.any(np.isnan(im)) or np.isinf(np.min(im)):
            im = np.nan_to_num(im, nan=np.nanmin(im[im != -np.inf]), posinf=np.nanmax(im[im != np.inf]), neginf=np.nanmin(im[im != -np.inf]))
        
        chunk = im[:, z_start : z_start + chunk_zlen, :, :]
        
        im_norm = im - chunk.mean()
        im_norm = im_norm / chunk.std(dtype=np.float64)
        
        return im_norm.astype(np.float32)

    def mean_norm(self, im: NdarrayOrTensor):
        m, s = norm.fit(im.flat)
        strech_min = max(m - self.lbound * s, im.min())
        strech_max = min(m + self.ubound * s, im.max())
        im[im > strech_max] = strech_max
        im[im < strech_min] = strech_min
        im = (im - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
        
        return im
        
    def __call__(self, img_dict: Dict[str, NdarrayOrTensor]) -> NdarrayOrTensor:
        
        for key in img_dict.keys():
            if key in self.keys:
                img_dict[key] = self.normalize(img_dict[key])
        
        return img_dict