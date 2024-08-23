from typing import Tuple

import numpy as np
from einops import rearrange
from monai.transforms import RandomizableTransform
from skimage.segmentation import find_boundaries

from cyto_dl.nn.vits.utils import match_tuple_dimensions


class JEPAMaskGenerator(RandomizableTransform):
    """Transform for generating Block-contiguous masks for JEPA training.

    This class works by randomly adding mask blocks until the mask_ratio is met or exceeded, then
    removing blocks from the exterior of the contiguous mask until the mask_ratio is met exactly.
    """

    def __init__(
        self,
        spatial_dims: int,
        mask_size: int = 12,
        block_aspect_ratio: Tuple[float] = (0.5, 1.5),
        num_patches: Tuple[float] = (6, 24, 24),
        mask_ratio: float = 0.9,
    ):
        """
        Parameters
        ----------
        spatial_dims : int
            The number of spatial dimensions of the image (2 or 3)
        mask_size : int, optional
            The size of the blocks used to generate mask. Block dimensions are determined by the mask size and an aspect ratio sampled from the range  `block_aspect_ratio`
        block_aspect_ratio : Tuple[float], optional
            The low and high values for aspect ratio of the mask blocks
        num_patches : Tuple[int], optional
            The number of patches used by the encoder for each dimension of the image (ZYX for 3D, YX for 2D)
        mask_ratio : float, optional
            The proportion of the image to be masked
        """
        assert 0 < mask_ratio < 1, "mask_ratio must be between 0 and 1"

        num_patches = match_tuple_dimensions(spatial_dims, [num_patches])[0]
        assert mask_size * max(block_aspect_ratio) < min(
            num_patches[-2:]
        ), "mask_size * max mask aspect ratio must be less than the smallest dimension of num_patches"

        self.mask_size = mask_size
        self.block_aspect_ratio = block_aspect_ratio
        self.num_patches = num_patches
        # convert mask_ratio to number of pixels to be masked
        self.target_pix = int(mask_ratio * np.prod(num_patches))

        self.mask = np.zeros(num_patches)
        self.edge_mask = np.ones(num_patches)

        self.spatial_dims = spatial_dims
        # create a mask that identified pixels on the edge of the image
        if self.spatial_dims == 3:
            self.edge_mask[1:-1, 1:-1, 1:-1] = 0
        elif self.spatial_dims == 2:
            self.edge_mask[1:-1, 1:-1] = 0
        else:
            raise ValueError("num_patches must be 2 or 3 dimensions")

    def remove_excess_pixels(self, mask):
        """Remove pixels along the boundary of the mask until the target number of pixels is
        reached."""
        bound = find_boundaries(mask, mode="inner")
        # include image edge as boundary, not just 1:0 transitions
        edge_mask = np.logical_and(mask, self.edge_mask)
        bound = np.logical_or(bound, edge_mask)
        bound_coords = np.argwhere(bound)
        # find number of pixels to remove from contiguous mask
        excess = int(mask.sum() - self.target_pix)
        remove = self.R.choice(range(bound_coords.shape[0]), excess, replace=False)
        remove_coords = bound_coords[remove]
        if self.spatial_dims == 3:
            mask[remove_coords[:, 0], remove_coords[:, 1], remove_coords[:, 2]] = 0
            mask = rearrange(mask, "z y x -> (z y x)").astype(bool)
        else:
            mask[remove_coords[:, 0], remove_coords[:, 1]] = 0
            mask = rearrange(mask, "y x -> (y x)").astype(bool)
        return mask

    def __call__(self, img_dict):
        # generate context (small) and target(large) masks
        # target: add blocks until target_pix is reached, then randomly remove excess border pixels
        # context: invert of target mask
        mask = self.mask.copy()
        while mask.sum() < self.target_pix:
            # randomly select block shape
            aspect_ratio = self.R.uniform(*self.block_aspect_ratio)
            width = int(self.mask_size * aspect_ratio)
            height = int(self.mask_size / aspect_ratio)
            # randomly select block position
            x = self.R.randint(0, self.num_patches[-1] - width + 1)
            y = self.R.randint(0, self.num_patches[-2] - height + 1)
            # add block to mask
            if self.spatial_dims == 3:
                mask[:, y : y + height, x : x + width] = 1
            else:
                mask[y : y + height, x : x + width] = 1

        mask = self.remove_excess_pixels(mask)

        context_mask = np.argwhere(~mask).squeeze()
        target_mask = np.argwhere(mask).squeeze()
        img_dict["context_mask"] = context_mask
        img_dict["target_mask"] = target_mask
        return img_dict
