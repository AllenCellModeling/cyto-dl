from typing import List, Optional

import numpy as np
from einops.layers.torch import Rearrange

from cyto_dl.nn.vits.blocks.patchify.patchify_base import PatchifyBase
from cyto_dl.nn.vits.utils import take_indexes


class Patchify(PatchifyBase):
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
        super().__init__(
            patch_size=patch_size,
            emb_dim=emb_dim,
            n_patches=n_patches,
            spatial_dims=spatial_dims,
            context_pixels=context_pixels,
            input_channels=input_channels,
            tasks=tasks,
            learnable_pos_embedding=learnable_pos_embedding,
        )

    @property
    def img2token(self):
        return self.create_img2token()

    def get_mask_args(self, mask_ratio):
        num_patches = np.prod(self.n_patches)
        n_visible_patches = int(num_patches * (1 - mask_ratio))
        return n_visible_patches, num_patches

    def create_img2token(self):
        """Rearranges the image tensor to a sequence of patches."""
        if self.spatial_dims == 3:
            return Rearrange("b c z y x -> (z y x) b c")
        elif self.spatial_dims == 2:
            return Rearrange("b c y x -> (y x) b c")

    def extract_visible_tokens(self, tokens, forward_indexes, n_visible_patches):
        return take_indexes(tokens, forward_indexes)[:n_visible_patches]
