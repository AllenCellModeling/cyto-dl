from typing import Sequence

import numpy as np
import torch
from einops import rearrange, repeat
from positional_encodings.torch_encodings import (
    PositionalEncoding2D,
    PositionalEncoding3D,
)
from timm.models.layers import trunc_normal_


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1]))


def get_positional_embedding(
    num_patches: Sequence[int], emb_dim: int, use_cls_token: bool = True, learnable: bool = True
):
    """Generate a positional embedding (with or without a cls token) for a given number of patches
    and embedding dimension.

    Can be either learnable or fixed.
    """
    if learnable:
        pe = torch.nn.Parameter(torch.zeros(np.prod(num_patches) + int(use_cls_token), 1, emb_dim))
        trunc_normal_(pe, std=0.02)
        return pe
    else:
        test_tensor = torch.ones(1, *num_patches, emb_dim)
        if len(num_patches) not in (2, 3):
            raise ValueError("Only 2d and 3d positional encodings are supported")
        if len(num_patches) == 2:
            pe = PositionalEncoding2D(emb_dim)(test_tensor)
            pe = rearrange(pe, "b y x c -> (y x) b c")
        elif len(num_patches) == 3:
            pe = PositionalEncoding3D(emb_dim)(test_tensor)
            pe = rearrange(pe, "b z y x c -> (z y x) b c")
        if use_cls_token:
            cls_token = torch.zeros(1, 1, emb_dim)
            pe = torch.cat([cls_token, pe], dim=0)
        return torch.nn.Parameter(pe, requires_grad=False)
