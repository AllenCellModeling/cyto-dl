# modified from https://github.com/IcarusWizard/MAE/blob/main/model.py#L124
# inspired by https://github.com/facebookresearch/hiera

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

from cyto_dl.nn.vits.blocks import IntermediateWeigher, Patchify
from cyto_dl.nn.vits.blocks.masked_unit_attention import HieraBlock
from cyto_dl.nn.vits.blocks.patchify import PatchifyHiera
from cyto_dl.nn.vits.utils import match_tuple_dimensions


class MAE_Encoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: List[int],
        spatial_dims: int = 3,
        patch_size: Union[int, List[int]] = 4,
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 12,
        num_head: Optional[int] = 3,
        context_pixels: Optional[Union[int, List[int]]] = 0,
        input_channels: Optional[int] = 1,
        n_intermediate_weights: Optional[int] = -1,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int], int
            Number of patches in each dimension. If a single int is provided, the number of patches in each dimension will be the same.
        spatial_dims: int
            Number of spatial dimensions
        patch_size: List[int]
            Size of each patch
        emb_dim: int
            Dimension of embedding
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        context_pixels: List[int], int
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension. If a single int is provided, the number of context pixels in each dimension will be the same.
        input_channels: int
            Number of input channels
        n_intermediate_weights: bool
            Whether to use intermediate weights for weighted sum of intermediate layers
        """
        super().__init__()
        num_patches, patch_size, context_pixels = match_tuple_dimensions(
            spatial_dims, [num_patches, patch_size, context_pixels]
        )

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.patchify = Patchify(
            patch_size, emb_dim, num_patches, spatial_dims, context_pixels, input_channels
        )
        weight_intermediates = n_intermediate_weights > 0
        if weight_intermediates:
            self.transformer = torch.nn.ModuleList(
                [Block(emb_dim, num_head) for _ in range(num_layer)]
            )
        else:
            self.transformer = torch.nn.Sequential(
                *[Block(emb_dim, num_head) for _ in range(num_layer)]
            )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.intermediate_weighter = (
            IntermediateWeigher(num_layer, emb_dim, n_intermediate_weights)
            if weight_intermediates
            else None
        )
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, img, mask_ratio=0.75):
        patches, mask, forward_indexes, backward_indexes = self.patchify(img, mask_ratio)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, "t b c -> b t c")

        if self.intermediate_weighter is not None:
            intermediates = [patches]
            for block in self.transformer:
                patches = block(patches)
                intermediates.append(patches)
            features = self.layer_norm(self.intermediate_weighter(intermediates))
            features = rearrange(features, "n b t c -> n t b c")
        else:
            features = self.layer_norm(self.transformer(patches))
            features = rearrange(features, "b t c -> t b c")
        if mask_ratio > 0:
            return features, mask, forward_indexes, backward_indexes
        return features


class JEPAEncoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: Union[int, List[int]],
        spatial_dims: int = 3,
        patch_size: Union[int, List[int]] = 4,
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 12,
        num_head: Optional[int] = 3,
        context_pixels: Optional[Union[int, List[int]]] = 0,
        input_channels: Optional[int] = 1,
        learnable_pos_embedding: Optional[bool] = True,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int], int
            Number of patches in each dimension. If a single int is provided, the number of patches in each dimension will be the same.
        spatial_dims: int
            Number of spatial dimensions
        patch_size: List[int]
            Size of each patch
        emb_dim: int
            Dimension of embedding
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        context_pixels: List[int], int
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension. If a single int is provided, the number of context pixels in each dimension will be the same.
        input_channels: int
            Number of input channels
        learnable_pos_embedding: bool
            If True, learnable positional embeddings are used. If False, fixed sin/cos positional embeddings. Empirically, fixed positional embeddings work better for brightfield images.
        """
        super().__init__()
        num_patches, patch_size, context_pixels = match_tuple_dimensions(
            spatial_dims, [num_patches, patch_size, context_pixels]
        )

        self.patchify = Patchify(
            patch_size=patch_size,
            emb_dim=emb_dim,
            n_patches=num_patches,
            spatial_dims=spatial_dims,
            context_pixels=context_pixels,
            input_channels=input_channels,
            learnable_pos_embedding=learnable_pos_embedding,
        )

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

    def forward(self, img, patchify=True):
        if patchify:
            patches, _, _, _ = self.patchify(img, mask_ratio=0)
        else:
            patches = img
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        return features


class SpatialMerger(nn.Module):
    """Class for converting multi-resolution Hiera features to the same (lowest) spatial resolution
    via convolution."""

    def __init__(
        self, downsample_factor: List[int], in_dim: int, out_dim: int, spatial_dims: int = 3
    ):
        super().__init__()
        downsample_factor = match_tuple_dimensions(spatial_dims, [downsample_factor])[0]

        self.spatial_dims = spatial_dims
        conv_fn = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        conv = conv_fn(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0,
            bias=False,
        )
        if spatial_dims == 3:
            tokens2img = Rearrange(
                "b n_mu (z y x) c -> (b n_mu) c z y x",
                z=downsample_factor[0],
                y=downsample_factor[1],
                x=downsample_factor[2],
            )
        else:
            tokens2img = Rearrange(
                "b n_mu (y x) c -> (b n_mu) c y x",
                y=downsample_factor[0],
                x=downsample_factor[1],
            )
        self.model = nn.Sequential(tokens2img, conv)

    def forward(self, x):
        b, n_mu, _, _ = x.shape
        x = self.model(x)
        if self.spatial_dims == 3:
            x = rearrange(x, "(b n_mu) c z y x -> b n_mu (z y x) c", b=b, n_mu=n_mu)
        else:
            x = rearrange(x, "(b n_mu) c y x -> b n_mu (y x) c", b=b, n_mu=n_mu)
        return x


class HieraEncoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: Union[int, List[int]],
        num_mask_units: Union[int, List[int]],
        architecture: List[Dict],
        emb_dim: int = 64,
        spatial_dims: int = 3,
        patch_size: Union[int, List[int]] = 4,
        context_pixels: Optional[Union[int, List[int]]] = 0,
        input_channels: Optional[int] = 1,
        save_layers: Optional[bool] = False,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: int, List[int]
            Number of patches in each dimension. If a single int is provided, the number of patches in each dimension will be the same.
        num_mask_units: int, List[int]
            Number of mask units in each dimension. If a single int is provided, the number of mask units in each dimension will be the same.
        architecture: List[Dict]
            List of dictionaries specifying the architecture of the transformer. Each dictionary should have the following keys:
            - repeat: int
                Number of times to repeat the block
            - num_heads: int
                Number of heads in the multihead attention
            - q_stride: int, List[int]
                Stride for the query in each spatial dimension
            - self_attention: bool
                Whether to use self attention or mask unit attention
            On the last repeat of each non-self-attention block, the embedding dimension is doubled and spatial pooling with `q_stride` is performed within each mask unit. For example, a block with a embed_dim=4, q_stride=2, and repeat=2, the first repeat just does mask unit attention, while the second will produce an 8-dimensional output that has been spatially pooled.
        emb_dim: int
            Dimension of embedding
        spatial_dims: int
            Number of spatial dimensions
        patch_size: List[int]
            Size of each patch
        context_pixels: List[int]
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension.
        input_channels: int
            Number of input channels
        save_layers: bool
            Whether to save the intermediate layer outputs
        """
        super().__init__()
        num_patches, num_mask_units, patch_size, context_pixels = match_tuple_dimensions(
            spatial_dims, [num_patches, num_mask_units, patch_size, context_pixels]
        )
        # make sure q stride shape matches spatial dims
        for i in range(len(architecture)):
            if "q_stride" in architecture[i]:
                architecture[i]["q_stride"] = match_tuple_dimensions(
                    spatial_dims, [architecture[i]["q_stride"]]
                )[0]

        self.save_layers = save_layers
        self.patchify = PatchifyHiera(
            patch_size=patch_size,
            n_patches=num_patches,
            emb_dim=emb_dim,
            spatial_dims=spatial_dims,
            context_pixels=context_pixels,
            input_channels=input_channels,
            mask_units_per_dim=num_mask_units,
        )

        patches_per_mask_unit = np.array(num_patches) // np.array(num_mask_units)

        total_downsampling_per_axis = np.prod(
            [block.get("q_stride", [1] * spatial_dims) for block in architecture], axis=0
        )

        assert np.all(
            patches_per_mask_unit - total_downsampling_per_axis >= 0
        ), f"Number of mask units must be greater than the total downsampling ratio, got {patches_per_mask_unit} patches per mask unit and {total_downsampling_per_axis} total downsampling ratio. Please adjust your q_stride or increase the number of patches per mask unit."
        assert np.all(
            patches_per_mask_unit % total_downsampling_per_axis == 0
        ), f"Number of mask units must be divisible by the total downsampling ratio, got {patches_per_mask_unit} patches per mask unit and {total_downsampling_per_axis} total downsampling ratio. Please adjust your q_stride"

        # number of output features doubles in each masked unit attention block, stays constant during self attention blocks
        self.final_dim = emb_dim * (2 ** len(architecture))

        self.save_block_idxs = []
        self.save_block_dims = []
        self.spatial_mergers = torch.nn.ParameterDict({})
        transformer = []
        num_blocks = 0
        for stage_num, stage in enumerate(architecture):
            # use mask unit attention until first layer that uses self attention
            if stage.get("self_attention", False):
                break
            print(f"Stage: {stage_num}")
            for block in range(stage["repeat"]):
                is_last = block == stage["repeat"] - 1
                # do spatial pooling within mask unit on last block of stage
                q_stride = stage["q_stride"] if is_last else [1] * spatial_dims

                # double embedding dimension in last block of stage
                dim_in = emb_dim * (2**stage_num)
                dim_out = dim_in if not is_last else dim_in * 2
                print(
                    f"\tBlock {block}:\t\tdim_in: {dim_in}, dim_out: {dim_out}, num_heads: {stage['num_heads']}, q_stride: {q_stride}, patches_per_mask_unit: {patches_per_mask_unit}"
                )
                transformer.append(
                    HieraBlock(
                        dim=dim_in,
                        dim_out=dim_out,
                        heads=stage["num_heads"],
                        spatial_dims=spatial_dims,
                        q_stride=q_stride,
                        patches_per_mask_unit=patches_per_mask_unit,
                    )
                )
                if is_last:
                    # save the block before the spatial pooling unless it's the final stage
                    save_block = (
                        num_blocks - 1 if stage_num < len(architecture) - 1 else num_blocks
                    )
                    self.save_block_idxs.append(save_block)
                    self.save_block_dims.append(dim_in)

                    # create a spatial merger for combining tokens pre-downsampling, last stage doesn't need merging since it has expected num channels, spatial shape
                    self.spatial_mergers[f"block_{save_block}"] = (
                        SpatialMerger(
                            patches_per_mask_unit,
                            dim_in,
                            self.final_dim,
                            spatial_dims=spatial_dims,
                        )
                        if stage_num < len(architecture) - 1
                        else torch.nn.Identity()
                    )

                    # at end of each layer, patches per mask unit is reduced as we pool spatially within mask units
                    patches_per_mask_unit = patches_per_mask_unit // np.array(stage["q_stride"])
                num_blocks += 1
        self.mask_unit_transformer = torch.nn.Sequential(*transformer)
        self.save_block_dims.append(self.final_dim)
        self.save_block_dims.reverse()

        self.self_attention_transformer = torch.nn.Sequential(
            *[Block(self.final_dim, stage["num_heads"]) for _ in range(stage["repeat"])]
        )

        self.layer_norm = torch.nn.LayerNorm(self.final_dim)

    def forward(self, img, mask_ratio):
        patches, mask, forward_indexes, backward_indexes = self.patchify(img, mask_ratio)

        # mask unit attention
        mask_unit_embeddings = 0.0
        save_layers = []
        for i, block in enumerate(self.mask_unit_transformer):
            patches = block(patches)
            if i in self.save_block_idxs:
                mask_unit_embeddings += self.spatial_mergers[f"block_{i}"](patches)
                if self.save_layers:
                    save_layers.append(patches)

        # combine mask units and tokens for full self attention transformer
        mask_unit_embeddings = rearrange(mask_unit_embeddings, "b n_mu t c -> b (n_mu t) c")
        mask_unit_embeddings = self.self_attention_transformer(mask_unit_embeddings)
        mask_unit_embeddings = self.layer_norm(mask_unit_embeddings)
        mask_unit_embeddings = rearrange(mask_unit_embeddings, "b t c -> t b c")

        return mask_unit_embeddings, mask, forward_indexes, backward_indexes  # , save_layers
