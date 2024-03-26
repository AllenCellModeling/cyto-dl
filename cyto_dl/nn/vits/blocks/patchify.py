from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_

from cyto_dl.nn.vits.utils import take_indexes


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


class Patchify(torch.nn.Module):
    """Class for converting images to a masked sequence of patches with positional embeddings."""

    def __init__(
        self,
        patch_size: List[int],
        emb_dim: int,
        n_patches: List[int],
        spatial_dims: int = 3,
        context_pixels: List[int] = [0, 0, 0],
        input_channels: int = 1,
        tasks: Optional[List[str]] = [],
    ):
        """
        Parameters
        ----------
        patch_size: List[int]
            Size of each patch
        emb_dim: int
            Dimension of encoder
        n_patches: List[int]
            Number of patches in each spatial dimension
        spatial_dims: int
            Number of spatial dimensions
        context_pixels: List[int]
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension.
        input_channels: int
            Number of input channels
        tasks: List[str]
            List of tasks to encode
        """
        super().__init__()
        self.n_patches = np.asarray(n_patches)
        self.spatial_dims = spatial_dims

        self.pos_embedding = torch.nn.Parameter(torch.zeros(np.prod(n_patches), 1, emb_dim))

        context_pixels = context_pixels[:spatial_dims]
        weight_size = np.asarray(patch_size) + np.round(np.array(context_pixels) * 2).astype(int)

        if spatial_dims == 3:
            self.conv = nn.Conv3d(
                in_channels=input_channels,
                out_channels=emb_dim,
                kernel_size=weight_size,
                stride=patch_size,
                padding=context_pixels,
            )
            self.img2token = Rearrange("b c z y x -> (z y x) b c")
            self.patch2img = Rearrange(
                "(n_patch_z n_patch_y n_patch_x) b c ->  b c n_patch_z n_patch_y n_patch_x",
                n_patch_z=n_patches[0],
                n_patch_y=n_patches[1],
                n_patch_x=n_patches[2],
            )
        elif spatial_dims == 2:
            self.conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=emb_dim,
                kernel_size=weight_size,
                stride=patch_size,
                padding=context_pixels,
            )
            self.img2token = Rearrange("b c y x -> (y x) b c")
            self.patch2img = Rearrange(
                "(n_patch_y n_patch_x) b c ->  b c n_patch_y n_patch_x",
                n_patch_y=n_patches[0],
                n_patch_x=n_patches[1],
            )

        self.task_embedding = torch.nn.ParameterDict(
            {task: torch.nn.Parameter(torch.zeros(1, 1, emb_dim)) for task in tasks}
        )
        self._init_weight()

    def _init_weight(self):
        trunc_normal_(self.pos_embedding, std=0.02)
        for task in self.task_embedding:
            trunc_normal_(self.task_embedding[task], std=0.02)

    def get_mask(self, img, n_visible_patches, num_patches):
        B = img.shape[0]

        indexes = [random_indexes(num_patches) for _ in range(B)]
        # forward indexes : index in image -> shuffledpatch
        forward_indexes = torch.as_tensor(
            np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long
        )
        # backward indexes : shuffled patch -> index in image
        backward_indexes = torch.as_tensor(
            np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long
        )

        mask = torch.zeros(num_patches, B, 1)
        # visible patches are first
        mask[:n_visible_patches] = 1
        mask = take_indexes(mask, backward_indexes)
        mask = self.patch2img(mask)
        # one pixel per masked patch, interpolate to size of input image
        mask = torch.nn.functional.interpolate(
            mask, img.shape[-self.spatial_dims :], mode="nearest"
        )

        return mask.to(img), forward_indexes, backward_indexes

    def forward(self, img, mask_ratio, task=None):
        # generate mask
        num_patches = np.prod(self.n_patches)
        n_visible_patches = int(num_patches * (1 - mask_ratio))
        mask, forward_indexes, backward_indexes = self.get_mask(
            img, n_visible_patches, num_patches
        )
        # generate patches
        tokens = self.conv(img * mask)
        tokens = self.img2token(tokens)
        # add position embedding
        tokens = tokens + self.pos_embedding
        if mask_ratio > 0:
            # extract visible patches
            tokens = take_indexes(tokens, forward_indexes)[:n_visible_patches]

        if task in self.task_embedding:
            tokens = tokens + self.task_embedding[task]

        # mask is used above to mask out patches, we need to invert it for loss calculation
        mask = (1 - mask).bool()

        return tokens, mask, forward_indexes, backward_indexes
