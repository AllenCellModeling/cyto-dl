from typing import List

import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

from cyto_dl.nn.vits.utils import random_indexes, take_indexes


def take_indexes_mask(sequences, indexes):
    """
    sequences: batch x mask units x patches x emb_dim
    indexes: mask_units x batch
    """
    # always gather across tokens dimension
    return torch.gather(
        sequences,
        1,
        repeat(
            indexes.to(sequences.device),
            "mu b -> b mu p c",
            b=sequences.shape[0],
            c=sequences.shape[-1],
            mu=sequences.shape[1],
            p=sequences.shape[2],
        ),
    )


class PatchifyHiera(torch.nn.Module):
    """Class for converting images to a masked sequence of patches with positional embeddings."""

    def __init__(
        self,
        patch_size: List[int],
        n_patches: List[int],
        mask_ratio: float = 0.8,
        num_mask_units: List[int] = [8, 8, 8],
        emb_dim: int = 64,
        spatial_dims: int = 3,
        context_pixels: List[int] = [0, 0, 0],
    ):
        """
        Parameters
        ----------
        patch_size: List[int]
            Size of each patch in pix (ZYX order for 3D, YX order for 2D)
        n_patches: List[int]
            Number of patches in each spatial dimension (ZYX order for 3D, YX order for 2D)
        mask_ratio: float
            Fraction of mask units to remove
        num_mask_units: List[int]
            Number of mask units in each spatial dimension (Z, Y, X)
        emb_dim: int
            Dimension of encoder
        spatial_dims: int
            Number of spatial dimensions
        context_pixels: List[int]
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.mask_ratio = mask_ratio
        self.total_n_mask_units = np.prod(num_mask_units)
        patches_per_mask_unit = np.prod(n_patches) // self.total_n_mask_units
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros(1, self.total_n_mask_units, patches_per_mask_unit, emb_dim)
        )

        self.num_mask_units = num_mask_units
        self.num_selected_mask_units = int(self.total_n_mask_units * (1 - mask_ratio))

        # mu -> mask unit
        self.mask2img = Rearrange(
            "(n_mu_z n_mu_y n_mu_x) b c -> b c  n_mu_z n_mu_y n_mu_x ",
            n_mu_z=num_mask_units[0],
            n_mu_y=num_mask_units[1],
            n_mu_x=num_mask_units[2],
        )

        self.img2mask_units = Rearrange(
            "b c (n_mu_z z) (n_mu_y y) (n_mu_x x) -> b (n_mu_z n_mu_y n_mu_x) (z y x) c ",
            n_mu_z=num_mask_units[0],
            n_mu_y=num_mask_units[1],
            n_mu_x=num_mask_units[2],
        )

        context_pixels = context_pixels[:spatial_dims]
        weight_size = np.asarray(patch_size) + np.round(np.array(context_pixels) * 2).astype(int)
        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=emb_dim,
            kernel_size=weight_size,
            stride=patch_size,
            padding=context_pixels,
        )

    def get_mask(self, img):
        B = img.shape[0]
        indexes = [random_indexes(self.total_n_mask_units, device=img.device) for _ in range(B)]
        # forward indexes : index in image -> shuffledpatch
        forward_indexes = torch.stack([i[0] for i in indexes], axis=-1)
        # backward indexes : shuffled patch -> index in image
        backward_indexes = torch.stack([i[1] for i in indexes], axis=-1)

        mask = torch.zeros(self.total_n_mask_units, B, 1, device=img.device, dtype=torch.uint8)
        # visible patches are first
        mask[: self.num_selected_mask_units] = 1
        mask = take_indexes(mask, backward_indexes)
        mask = self.mask2img(mask)
        # one pixel per masked patch, interpolate to size of input image
        mask = torch.nn.functional.interpolate(
            mask, img.shape[-self.spatial_dims :], mode="nearest"
        )
        return mask, forward_indexes, backward_indexes

    def forward(self, img):
        """" takes in BCZYX image returns B x num_selected_mask_units x patches_per_mask_unit x
        emb_dim."""
        mask = torch.ones_like(img)
        forward_indexes, backward_indexes = None, None
        if self.mask_ratio > 0:
            mask, forward_indexes, backward_indexes = self.get_mask(img)
        tokens = self.conv(img * mask)
        # break into batch x mask units x patches permask unit x emb_dim
        tokens = self.img2mask_units(tokens)

        tokens = tokens + self.pos_embedding
        if self.mask_ratio > 0:
            tokens = take_indexes_mask(tokens, forward_indexes)[:, : self.num_selected_mask_units]
        mask = (1 - mask).bool()
        return tokens, mask, forward_indexes, backward_indexes
