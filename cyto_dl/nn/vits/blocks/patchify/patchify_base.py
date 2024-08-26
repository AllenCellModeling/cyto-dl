from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_

from cyto_dl.nn.vits.utils import get_positional_embedding, random_indexes, take_indexes


class PatchifyBase(torch.nn.Module, ABC):
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
        learnable_pos_embedding: bool = True,
    ):
        """
        Parameters
        ----------
        patch_size: List[int]
            Size of each patch in pix (ZYX order for 3D, YX order for 2D)
        emb_dim: int
            Dimension of encoder
        n_patches: List[int]
            Number of patches in each spatial dimension (ZYX order for 3D, YX order for 2D)
        spatial_dims: int
            Number of spatial dimensions
        context_pixels: List[int]
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension.
        input_channels: int
            Number of input channels
        tasks: List[str]
            List of tasks to encode
        learnable_pos_embedding: bool
            If True, learnable positional embeddings are used. If False, fixed sin/cos positional embeddings. Empirically, fixed positional embeddings work better for brightfield images.
        """
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("Only 2D and 3D images are supported")
        self.spatial_dims = spatial_dims
        self.n_patches = np.asarray(n_patches)

        self.pos_embedding = get_positional_embedding(
            n_patches, emb_dim, learnable=learnable_pos_embedding, use_cls_token=False
        )

        self.patch2img = self.create_patch2img(n_patches, patch_size)
        self.conv = self.create_conv(input_channels, emb_dim, patch_size, context_pixels)

        self.task_embedding = torch.nn.ParameterDict(
            {task: torch.nn.Parameter(torch.zeros(1, 1, emb_dim)) for task in tasks}
        )
        self._init_weight()

    def _init_weight(self):
        for task in self.task_embedding:
            trunc_normal_(self.task_embedding[task], std=0.02)

    @property
    @abstractmethod
    def img2token(self):
        pass

    @abstractmethod
    def get_mask_args(self):
        pass

    @abstractmethod
    def extract_visible_tokens(self):
        pass

    def create_conv(self, input_channels, emb_dim, patch_size, context_pixels):
        context_pixels = context_pixels[: self.spatial_dims]
        weight_size = np.asarray(patch_size) + np.round(np.array(context_pixels) * 2).astype(int)

        if self.spatial_dims == 3:
            return nn.Conv3d(
                in_channels=input_channels,
                out_channels=emb_dim,
                kernel_size=weight_size,
                stride=patch_size,
                padding=context_pixels,
            )
        elif self.spatial_dims == 2:
            return nn.Conv2d(
                in_channels=input_channels,
                out_channels=emb_dim,
                kernel_size=weight_size,
                stride=patch_size,
                padding=context_pixels,
            )

    def create_patch2img(self, n_patches, patch_size):
        """Converts boolean array of whether to keep index of each patch to an image-shaped mask of
        same size as input image."""
        if self.spatial_dims == 3:
            return torch.nn.Sequential(
                *[
                    # rearrange tokens to image
                    Rearrange(
                        "(n_patch_z n_patch_y n_patch_x) b c ->  b c n_patch_z n_patch_y n_patch_x",
                        n_patch_z=n_patches[0],
                        n_patch_y=n_patches[1],
                        n_patch_x=n_patches[2],
                    ),
                    # nearest neighbor resize image to match input image size
                    Reduce(
                        "b c n_patch_z n_patch_y n_patch_x -> b c (n_patch_z patch_size_z) (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
                        reduction="repeat",
                        patch_size_z=patch_size[0],
                        patch_size_y=patch_size[1],
                        patch_size_x=patch_size[2],
                    ),
                ]
            )
        elif self.spatial_dims == 2:
            return torch.nn.Sequential(
                *[
                    Rearrange(
                        "(n_patch_y n_patch_x) b c ->  b c n_patch_y n_patch_x",
                        n_patch_y=n_patches[0],
                        n_patch_x=n_patches[1],
                    ),
                    Reduce(
                        "b c n_patch_y n_patch_x -> b c (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
                        reduction="repeat",
                        patch_size_y=patch_size[0],
                        patch_size_x=patch_size[1],
                    ),
                ]
            )

    def get_mask(self, img, n_visible_patches, num_patches):
        B = img.shape[0]

        indexes = [random_indexes(num_patches, img.device) for _ in range(B)]
        # forward indexes : index in image -> shuffledpatch
        forward_indexes = torch.stack([i[0] for i in indexes], axis=-1)

        # backward indexes : shuffled patch -> index in image
        backward_indexes = torch.stack([i[1] for i in indexes], axis=-1)

        mask = torch.zeros(num_patches, B, 1, device=img.device, dtype=torch.bool)
        # visible patches are first
        mask[:n_visible_patches] = True
        mask = take_indexes(mask, backward_indexes)
        mask = self.patch2img(mask)

        return mask, forward_indexes, backward_indexes

    def forward(self, img, mask_ratio, task=None):
        # generate mask
        mask = torch.ones_like(img).bool()
        forward_indexes, backward_indexes = None, None
        if mask_ratio > 0:
            n_visible_patches, num_patches = self.get_mask_args(mask_ratio)
            mask, forward_indexes, backward_indexes = self.get_mask(
                img, n_visible_patches, num_patches
            )
        # generate patches
        tokens = self.conv(img * mask)
        tokens = self.img2token(tokens)

        # add position embedding
        tokens = tokens + self.pos_embedding

        # extract visible patches
        if mask_ratio > 0:
            tokens = self.extract_visible_tokens(tokens, forward_indexes, n_visible_patches)

        # add task embedding
        if task in self.task_embedding:
            tokens = tokens + self.task_embedding[task]

        # mask is used above to mask out patches, we need to invert it for loss calculation
        return tokens, ~mask, forward_indexes, backward_indexes
