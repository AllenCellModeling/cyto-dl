# inspired by https://github.com/facebookresearch/hiera/tree/main
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Reduce
from timm.models.layers import DropPath, Mlp

from cyto_dl.nn.vits.utils import match_tuple_dimensions


class MaskUnitAttention(torch.nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        spatial_dims: int = 3,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        q_stride=[1, 1, 1],
        patches_per_mask_unit=[2, 12, 12],
    ):
        """
        Parameters
        ----------
        dim : int
            Dimension of the input features.
        dim_out : int
            Dimension of the output features.
        spatial_dims : int, optional
            Number of spatial dimensions, by default 3.
        num_heads : int, optional
            Number of attention heads, by default 8.
        qkv_bias : bool, optional
            If True, add a learnable bias to query, key, value, by default False.
        attn_drop : float, optional
            Dropout rate for attention, by default 0.0.
        proj_drop : float, optional
            Dropout rate for projection, by default 0.0.
        q_stride : List[int], optional
            Stride for query, by default [1, 1, 1].
        patches_per_mask_unit : List[int], optional
            Number of patches per mask unit, by default [2, 12, 12].
        """
        super().__init__()
        q_stride, patches_per_mask_unit = match_tuple_dimensions(
            spatial_dims, [q_stride, patches_per_mask_unit]
        )

        self.spatial_dims = spatial_dims
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dim_out = dim_out
        self.q_stride = np.array(q_stride)
        self.pooled_patches_per_mask_unit = (
            np.array(patches_per_mask_unit) / self.q_stride
        ).astype(int)

    def forward(self, x):
        # project and split into q,k,v embeddings
        qkv = rearrange(
            self.qkv(x),
            "batch num_mask_units tokens_per_mask_unit (head_dim num_heads qkv) -> qkv batch num_mask_units num_heads tokens_per_mask_unit head_dim",
            head_dim=self.head_dim,
            qkv=3,
            num_heads=self.num_heads,
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        if np.any(self.q_stride > 1):
            # within a mask unit, tokens are spatially ordered
            # perform spatial 2x2x2 max pooling over tokens
            if self.spatial_dims == 3:
                q = reduce(
                    q,
                    "batch num_mask_units num_heads (n_patches_z q_stride_z n_patches_y q_stride_y n_patches_x q_stride_x) c -> batch num_mask_units num_heads (n_patches_z n_patches_y n_patches_x) c",
                    reduction="max",
                    q_stride_z=self.q_stride[0],
                    q_stride_y=self.q_stride[1],
                    q_stride_x=self.q_stride[2],
                    n_patches_z=self.pooled_patches_per_mask_unit[0],
                    n_patches_y=self.pooled_patches_per_mask_unit[1],
                    n_patches_x=self.pooled_patches_per_mask_unit[2],
                )
            elif self.spatial_dims == 2:
                q = reduce(
                    q,
                    "batch num_mask_units num_heads (n_patches_y q_stride_y n_patches_x q_stride_x) c ->batch num_mask_units num_heads (n_patches_y n_patches_x) c",
                    reduction="max",
                    q_stride_y=self.q_stride[0],
                    q_stride_x=self.q_stride[1],
                    n_patches_y=self.pooled_patches_per_mask_unit[0],
                    n_patches_x=self.pooled_patches_per_mask_unit[1],
                )

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop,
        )
        # combine heads into single channel dimension
        x = rearrange(
            attn, "b mask_units n_heads t c -> b mask_units t (n_heads c)", n_heads=self.num_heads
        )

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        spatial_dims: int = 3,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        q_stride: List[int] = [1, 1, 1],
        patches_per_mask_unit: List[int] = [2, 12, 12],
    ):
        """
        Parameters
        ----------
        dim : int
            Dimension of the input features.
        dim_out : int
            Dimension of the output features.
        heads : int
            Number of attention heads.
        spatial_dims : int, optional
            Number of spatial dimensions, by default 3.
        mlp_ratio : float, optional
            Ratio of MLP hidden dim to embedding dim, by default 4.0.
        drop_path : float, optional
            Dropout rate for the path, by default 0.0.
        norm_layer : nn.Module, optional
            Normalization layer, by default nn.LayerNorm.
        act_layer : nn.Module, optional
            Activation layer for the MLP, by default nn.GELU.
        q_stride : List[int], optional
            Stride for query, by default [1, 1, 1].
        patches_per_mask_unit : List[int], optional
            Number of patches per mask unit, by default [2, 12, 12].
        """
        super().__init__()
        patches_per_mask_unit, q_stride = match_tuple_dimensions(
            spatial_dims, [patches_per_mask_unit, q_stride]
        )

        self.spatial_dims = spatial_dims
        self.dim = dim
        self.dim_out = dim_out
        self.q_stride = q_stride

        self.norm1 = norm_layer(dim)

        do_pool = np.any(np.array(q_stride) > 1) or dim != dim_out

        self.attn = MaskUnitAttention(
            dim,
            dim_out,
            spatial_dims=spatial_dims,
            num_heads=heads,
            q_stride=q_stride,
            patches_per_mask_unit=patches_per_mask_unit,
        )

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # mean pooling by q stride within a mask unit
        if self.spatial_dims == 3:
            skip_connection_pooling = Reduce(
                "b n (n_patches_z q_stride_z n_patches_y q_stride_y n_patches_x q_stride_x) c -> b n (n_patches_z n_patches_y n_patches_x) c",
                reduction="mean",
                q_stride_z=self.q_stride[0],
                q_stride_y=self.q_stride[1],
                q_stride_x=self.q_stride[2],
                n_patches_z=self.attn.pooled_patches_per_mask_unit[0],
                n_patches_y=self.attn.pooled_patches_per_mask_unit[1],
                n_patches_x=self.attn.pooled_patches_per_mask_unit[2],
            )
        elif self.spatial_dims == 2:
            skip_connection_pooling = Reduce(
                "b n (n_patches_y q_stride_y n_patches_x q_stride_x) c -> b n (n_patches_y n_patches_x) c",
                reduction="mean",
                q_stride_y=self.q_stride[0],
                q_stride_x=self.q_stride[1],
                n_patches_y=self.attn.pooled_patches_per_mask_unit[0],
                n_patches_x=self.attn.pooled_patches_per_mask_unit[1],
            )

        self.proj = (
            torch.nn.Sequential(skip_connection_pooling, nn.Linear(dim, dim_out))
            if do_pool
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: batch x mask units x tokens x emb_dim
        """
        # Attention + Q Pooling
        x_norm = self.norm1(x)

        # change dimension and subsample within mask unit for skip connection
        x = self.proj(x_norm)

        x = x + self.drop_path(self.attn(x_norm))
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
