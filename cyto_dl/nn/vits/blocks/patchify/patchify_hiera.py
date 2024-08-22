from typing import List, Optional

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_

from cyto_dl.nn.vits.blocks.patchify.patchify_base import PatchifyBase


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
            indexes,
            "mu b -> b mu p c",
            b=sequences.shape[0],
            c=sequences.shape[-1],
            mu=sequences.shape[1],
            p=sequences.shape[2],
        ),
    )


class PatchifyHiera(PatchifyBase):
    """Class for converting images to a sequence of patches with positional embeddings, masked at
    the level of mask units (groups of patches specified by mask_units_per_dim)."""

    def __init__(
        self,
        patch_size: List[int],
        n_patches: List[int],
        emb_dim: int = 64,
        spatial_dims: int = 3,
        context_pixels: List[int] = [0, 0, 0],
        input_channels: int = 1,
        tasks: Optional[List[str]] = [],
        mask_units_per_dim: List[int] = [8, 8, 8],
    ):
        """
        patch_size: List[int]
            Size of each patch in pix (ZYX order for 3D, YX order for 2D)
        n_patches: List[int]
            Number of patches in each spatial dimension (ZYX order for 3D, YX order for 2D)
        emb_dim: int
            Dimension of encoder
        spatial_dims: int
            Number of spatial dimensions
        context_pixels: List[int]
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension.
        input_channels: int
            Number of input channels
        tasks: List[str]
            List of tasks to encode
        mask_units_per_dim: List[int]
            Number of mask units in each spatial dimension (ZYX order for 3D, YX order for 2D)
        """
        super().__init__(
            patch_size=patch_size,
            emb_dim=emb_dim,
            n_patches=n_patches,
            spatial_dims=spatial_dims,
            context_pixels=context_pixels,
            input_channels=input_channels,
            tasks=tasks,
            learnable_pos_embedding=True,
        )

        self.total_n_mask_units = np.prod(mask_units_per_dim)
        # mask_unit_size is the img shape / mask_units_per_dim, img_shape is size per patch * n_patches
        mask_unit_size_pix = (
            (np.array(patch_size) * np.array(n_patches)) / np.array(mask_units_per_dim)
        ).astype(int)

        patches_per_mask_unit = mask_unit_size_pix // patch_size

        # rearrange patch embeddings to mask units
        self.pos_embedding = torch.nn.Parameter(
            rearrange(
                self.pos_embedding,
                "(ppmu total_n_mu) 1 emb_dim -> 1 total_n_mu ppmu emb_dim",
                total_n_mu=self.total_n_mask_units,
                ppmu=patches_per_mask_unit.prod(),
                emb_dim=emb_dim,
            )
        )

        self.mask_units_per_dim = mask_units_per_dim

        self.patch2img = self.create_patch2img(mask_units_per_dim, mask_unit_size_pix)

    @property
    def img2token(self):
        # redefine this to work with mask units instead of patches
        return self.create_img2token(self.mask_units_per_dim)

    # in hiera, the masking is done at the mask unit, not the patch level
    def get_mask_args(self, mask_ratio):
        n_visible_patches = int(self.total_n_mask_units * (1 - mask_ratio))
        return n_visible_patches, self.total_n_mask_units

    def create_img2token(self, mask_units_per_dim):
        if self.spatial_dims == 3:
            return Rearrange(
                "b c (n_mu_z z) (n_mu_y y) (n_mu_x x) -> b (n_mu_z n_mu_y n_mu_x) (z y x) c ",
                n_mu_z=mask_units_per_dim[0],
                n_mu_y=mask_units_per_dim[1],
                n_mu_x=mask_units_per_dim[2],
            )
        elif self.spatial_dims == 2:
            return Rearrange(
                "b c  (n_mu_y y) (n_mu_x x) -> b (n_mu_y n_mu_x) (y x) c ",
                n_mu_y=mask_units_per_dim[0],
                n_mu_x=mask_units_per_dim[1],
            )

    def extract_visible_tokens(self, tokens, forward_indexes, n_visible_patches):
        return take_indexes_mask(tokens, forward_indexes)[:, :n_visible_patches]
