from monai.transforms import RandomizableTransform
import numpy as np
from skimage.segmentation import find_boundaries
from einops import rearrange
from typing import Tuple


class JEPAMaskGenerator(RandomizableTransform):
    def __init__(self, mask_size:int=12, mask_aspect_ratio: Tuple[float]=(0.5,1.5), num_patches: Tuple[float]=(6, 24, 24), mask_ratio: float=0.9):
        assert mask_ratio < 1, "mask_ratio must be less than 1"
        assert mask_size * max(mask_aspect_ratio) < min(num_patches[-2:]), "mask_size * max mask aspect ratio must be less than the smallest dimension of num_patches"
 
        self.mask_size = mask_size
        self.mask_aspect_ratio = mask_aspect_ratio
        self.num_patches = num_patches
        self.target_pix = int(mask_ratio * np.prod(num_patches))
        self.mask = np.zeros(num_patches)
        self.edge_mask = np.ones(num_patches)

        self.spatial_dims = len(num_patches)
        if self.spatial_dims == 3:
            self.edge_mask[1:-1, 1:-1, 1:-1] = 0
        elif self.spatial_dims == 2:
            self.edge_mask[1:-1, 1:-1] = 0
        else:
            raise ValueError("num_patches must be 2 or 3 dimensions")

    def __call__(self, img_dict):
        # generate context (small) and target(large) masks
        # target: add blocks until target_pix is reached, then randomly remove excess border pixels
        # context: invert of target mask
        mask = self.mask.copy()
        while mask.sum() < self.target_pix:
            aspect_ratio = np.random.uniform(*self.mask_aspect_ratio)
            width = int(self.mask_size*aspect_ratio)
            height = int(self.mask_size/aspect_ratio)
            x = np.random.randint(0, self.num_patches[-1]-width+1)
            y = np.random.randint(0, self.num_patches[-2]-height+1)
            if self.spatial_dims == 3:
                mask[:, y:y+height, x:x+width] = 1
            else:
                mask[y:y+height, x:x+width] = 1

        bound = find_boundaries(mask, mode='inner')
        # include image edge as boundary, not just 1:0 transitions
        edge_mask = np.logical_and(mask, self.edge_mask)

        bound = np.logical_or(bound, edge_mask)
        bound_coords = np.argwhere(bound)
        excess = int(mask.sum() - self.target_pix)
        remove = np.random.choice(range(bound_coords.shape[0]), excess, replace=False)
        remove_coords = bound_coords[remove]
        if self.spatial_dims == 3:
            mask[remove_coords[:, 0], remove_coords[:, 1], remove_coords[:, 2]] = 0
            mask = rearrange(mask, 'z y x -> (z y x)').astype(bool)
        else:
            mask[remove_coords[:, 0], remove_coords[:, 1]] = 0
            mask = rearrange(mask, 'y x -> (y x)').astype(bool)
        context_mask = np.argwhere(~mask).squeeze()
        target_mask = np.argwhere(mask).squeeze()
        img_dict['context_mask'] = context_mask
        img_dict['target_mask'] = target_mask
        return img_dict