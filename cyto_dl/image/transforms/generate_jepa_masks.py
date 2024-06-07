from monai.transforms import RandomizableTransform
import numpy as np
from skimage.segmentation import find_boundaries
from einops import rearrange
from typing import Tuple


class JEPAMaskGenerator(RandomizableTransform):
    def __init__(self, num_masks:int=4, mask_size:int=12, mask_aspect_ratio: Tuple[float]=(0.5,1.5), num_patches: Tuple[float]=(6, 24, 24), mask_ratio: float=0.9):
        self.num_masks = num_masks
        self.mask_size = mask_size
        self.mask_aspect_ratio = mask_aspect_ratio
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.target_pix = int(mask_ratio * np.prod(num_patches))
        self.num_patches = num_patches

    def __call__(self, img_dict):
        # generate context (small) and target(large) masks
        mask = np.zeros(self.num_patches)
        while mask.sum() < self.target_pix:
            aspect_ratio = np.random.uniform(*self.mask_aspect_ratio)
            width = int(self.mask_size*aspect_ratio)
            height = int(self.mask_size/aspect_ratio)
            x = np.random.randint(0, self.num_patches[2]-width)
            y = np.random.randint(0, self.num_patches[1]-height)
            mask[:, y:y+height, x:x+width] = 1
        bound = find_boundaries(mask, mode='inner')
        bound_coords = np.argwhere(bound)
        excess = int(mask.sum() - self.target_pix)
        remove = np.random.choice(range(bound_coords.shape[0]), excess, replace=False)
        remove_coords = bound_coords[remove]
        mask[remove_coords[:, 0], remove_coords[:, 1], remove_coords[:, 2]] = 0
        mask = rearrange(mask, 'z y x -> (z y x)').astype(bool)
        context_mask = np.argwhere(~mask).squeeze()
        target_mask = np.argwhere(mask).squeeze()
        img_dict['context_mask'] = context_mask
        img_dict['target_mask'] = target_mask
        return img_dict