from re import A
import aicssegmentation
import numpy as np
from monai.transforms.transform import Transform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from scipy.stats import norm
import torch
from typing import List, Dict, Union, Tuple
from skimage.exposure import cumulative_distribution as cdf
from skimage.exposure import rescale_intensity
from pathlib import Path
from glob import glob
from random import sample, seed, random
from multiprocessing import Pool
from aicsimageio import AICSImage
from monai.data import MetaTensor
from torch import Tensor


class MeanNormalizeIntensity(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self, 
            keys: List[str], 
            lower_bound: float, 
            upper_bound: float
        ) -> None:
        super().__init__()
        self.keys = keys
        self.lbound = lower_bound
        self.ubound = upper_bound
        
    def normalize(self, im: NdarrayOrTensor):
        
        is_tensor = not isinstance(im, np.ndarray)
        if is_tensor:
            im = np.array(im)
            
        m, s = norm.fit(im.flat)
        strech_min = max(m - self.lbound * s, im.min())
        strech_max = min(m + self.ubound * s, im.max())
        im[im > strech_max] = strech_max
        im[im < strech_min] = strech_min
        im = (im - strech_min + 1e-8)/(strech_max - strech_min + 1e-8)
        
        if is_tensor:
            im = torch.from_numpy(im)
        return im
        
    def __call__(self, img_dict: Dict[str, NdarrayOrTensor]) -> NdarrayOrTensor:
        
        for key in img_dict.keys():
            if key in self.keys:
                img_dict[key] = self.normalize(img_dict[key])
        
        return img_dict
    
class MatchHistogramToReference(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
            self, 
            keys: List[str], 
            reference: str, 
            file_format: Union[List[str], str] = '.tiff', 
            sample_size: Union[float, int] = 5,
            rand_seed: int = -1,
            prob: float = 0.5
        ) -> None:
        super().__init__()
        
        self.keys = keys
        self.prob = prob
        
        #currently transform only compatible with uint16 images
        self.nbins = 2**16
        self.hist_range = (0, 2**16-1)
        self.bins = np.arange(0, 2**16, 1)
        
        self.reference_cdf = self.calculate_cumulative_cdf(reference, file_format, sample_size, rand_seed)
        
    def calculate_cumulative_cdf(
            self, 
            reference: str, 
            file_format: Union[List[str], str], 
            sample_size: Union[float, int],
            rand_seed: int
        ) -> Dict[str, List[Union[int,float]]]:
        
        # if reference path is directory, sample files of file_format type from directory
        if Path(reference).is_dir():
            dir_files = []
            if not isinstance(file_format, list):
                file_format = [file_format]
                
            for ftype in file_format:
                dir_files += glob(reference + '/*' + ftype)
                
            if sample_size > 0:
                if sample_size < 1:
                    n = int(len(dir_files)*sample_size)
                    if n == 0:
                        n = 1
                else:
                    n = int(sample_size)
                
                if rand_seed > 0:
                    seed(rand_seed)
                reference_files = sample(dir_files, n)
            else:
                reference_files - dir_files
        else:
            reference_files = [reference]
            
        # calculate histograms for each reference image
        with Pool() as p:
            hists = p.map(self.image2hist, reference_files)
            
        # calculate total cdf of all reference images
        cumhist = np.sum(np.stack(hists,axis=-1),axis=-1)
        return np.cumsum(cumhist / np.sum(cumhist))
        
    def image2hist(self, fn: str):
        return self.calculate_histogram(AICSImage(fn).dask_data)
    
    def calculate_histogram(self, img: np.ndarray):
        hist, _ = np.histogram(img, bins=self.nbins, range=self.hist_range)
        return hist
    
    def calculate_cdf(self, img):
        hist = self.calculate_histogram(img)
        return np.cumsum(hist / np.sum(hist))
    
    def match_histogram(self, img):
        is_tensor = isinstance(img, Tensor)
        is_meta = isinstance(img, MetaTensor)
        if is_tensor:
            img_type = img.dtype
        if is_tensor or is_meta:
            img = np.array(img)
        
        
        cdf = self.calculate_cdf(img)
        interp_values = np.interp(cdf, self.reference_cdf, self.bins)
        
        def match_hist(i):
            # import pdb; pdb.set_trace()
            return interp_values[int(i)]
        match_all = np.vectorize(match_hist)
        
        img = match_all(img.flatten()).reshape(img.shape)
        if is_tensor:
            img = torch.from_numpy(img).type(img_type)
        if is_meta:
            img = MetaTensor(img)
            
        return img
    
    def __call__(self, img_dict: Dict[str, NdarrayOrTensor]) -> NdarrayOrTensor:
        
        for key in img_dict.keys():
            if key in self.keys:
                if random() < self.prob:
                    img_dict[key] = self.match_histogram(img_dict[key])
        
        return img_dict