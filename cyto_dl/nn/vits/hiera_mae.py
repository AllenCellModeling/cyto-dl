# inspired by https://github.com/facebookresearch/hiera

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Block

from cyto_dl.nn.vits.blocks.masked_unit_attention import HieraBlock
from cyto_dl.nn.vits.blocks.patchify_hiera import PatchifyHiera
from cyto_dl.nn.vits.cross_mae import CrossMAE_Decoder
from cyto_dl.nn.vits.mae import MAE_Decoder


class SpatialMerger(nn.Module):
    def __init__(self, downsample_factor, in_dim, out_dim):
        super().__init__()
        self.downsample_factor = downsample_factor
        conv = nn.Conv3d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0,
            bias=False,
        )

        tokens2img = Rearrange(
            "b n_mu (z y x) c -> (b n_mu) c z y x",
            z=downsample_factor[0],
            y=downsample_factor[1],
            x=downsample_factor[2],
        )
        self.model = nn.Sequential(tokens2img, conv)

    def forward(self, x):
        b, n_mu, _, _ = x.shape
        x = self.model(x)
        x = rearrange(x, "(b n_mu) c z y x -> b n_mu (z y x) c", b=b, n_mu=n_mu)
        return x


class HieraEncoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: List[int],
        num_mask_units: List[int],
        architecture: List[Dict],
        emb_dim: int = 64,
        spatial_dims: int = 3,
        patch_size: List[int] = (16, 16, 16),
        mask_ratio: Optional[float] = 0.75,
        context_pixels: Optional[List[int]] = [0, 0, 0],
        save_layers: Optional[bool] = True,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int]
            Number of patches in each dimension
        num_mask_units: List[int]
            Number of mask units in each dimension
        architecture: List[Dict]
            List of dictionaries specifying the architecture of the transformer. Each dictionary should have the following keys:
            - repeat: int
                Number of times to repeat the block
            - num_heads: int
                Number of heads in the multihead attention
            - q_stride: List[int]
                Stride for the query in each spatial dimension
            - self_attention: bool
                Whether to use self attention or mask unit attention
        emb_dim: int
            Dimension of embedding
        spatial_dims: int
            Number of spatial dimensions
        patch_size: List[int]
            Size of each patch
        mask_ratio: float
            Fraction of mask units to remove
        context_pixels: List[int]
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension.
        save_layers: bool
            Whether to save the intermediate layer outputs
        """
        super().__init__()
        self.save_layers = save_layers
        self.patchify = PatchifyHiera(
            patch_size,
            num_patches,
            mask_ratio,
            num_mask_units,
            emb_dim,
            spatial_dims,
            context_pixels,
        )

        patches_per_mask_unit = np.array(num_patches) // np.array(num_mask_units)
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
                        SpatialMerger(patches_per_mask_unit, dim_in, self.final_dim)
                        if stage_num < len(architecture) - 1
                        else torch.nn.Identity()
                    )

                    # at end of each layer, patches per mask unit is reduced as we pool spatially
                    patches_per_mask_unit = patches_per_mask_unit // np.array(stage["q_stride"])
                num_blocks += 1
        self.mask_unit_transformer = torch.nn.Sequential(*transformer)
        self.save_block_dims.append(self.final_dim)
        self.save_block_dims.reverse()

        self.self_attention_transformer = torch.nn.Sequential(
            *[Block(self.final_dim, stage["num_heads"]) for _ in range(stage["repeat"])]
        )

        self.layer_norm = torch.nn.LayerNorm(self.final_dim)

    def forward(self, img):
        patches, mask, forward_indexes, backward_indexes = self.patchify(img)

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

        return mask_unit_embeddings, mask, forward_indexes, backward_indexes, save_layers


class HieraMAE(torch.nn.Module):
    def __init__(
        self,
        architecture: List[Dict],
        spatial_dims: int = 3,
        num_patches: Optional[List[int]] = [2, 32, 32],
        num_mask_units: Optional[List[int]] = [2, 12, 12],
        patch_size: Optional[List[int]] = [16, 16, 16],
        emb_dim: Optional[int] = 64,
        decoder_layer: Optional[int] = 4,
        decoder_head: Optional[int] = 8,
        decoder_dim: Optional[int] = 192,
        mask_ratio: Optional[int] = 0.75,
        context_pixels: Optional[List[int]] = [0, 0, 0],
        use_crossmae: Optional[bool] = False,
    ) -> None:
        """
        Parameters
        ----------
        architecture: List[Dict]
            List of dictionaries specifying the architecture of the transformer. Each dictionary should have the following keys:
            - repeat: int
                Number of times to repeat the block
            - num_heads: int
                Number of heads in the multihead attention
            - q_stride: List[int]
                Stride for the query in each spatial dimension
            - self_attention: bool
                Whether to use self attention or mask unit attention
        spatial_dims: int
            Number of spatial dimensions
        num_patches: List[int]
            Number of patches in each dimension
        num_mask_units: List[int]
            Number of mask units in each dimension
        patch_size: List[int]
            Size of each patch
        emb_dim: int
            Dimension of embedding
        decoder_layer: int
            Number of layers in the decoder
        decoder_head: int
            Number of heads in the decoder
        decoder_dim: int
            Dimension of the decoder
        mask_ratio: float
            Fraction of mask units to remove
        context_pixels: List[int]
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension.
        """
        super().__init__()
        assert spatial_dims in (2, 3), "Spatial dims must be 2 or 3"

        if isinstance(num_patches, int):
            num_patches = [num_patches] * spatial_dims
        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial_dims

        assert len(num_patches) == spatial_dims, "num_patches must be of length spatial_dims"
        assert len(patch_size) == spatial_dims, "patch_size must be of length spatial_dims"

        self.mask_ratio = mask_ratio

        self.encoder = HieraEncoder(
            num_patches=num_patches,
            num_mask_units=num_mask_units,
            architecture=architecture,
            emb_dim=emb_dim,
            spatial_dims=spatial_dims,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            context_pixels=context_pixels,
        )
        # "patches" to the decoder are actually mask units, so num_patches is num_mask_units, patch_size is mask unit size
        mask_unit_size = (np.array(num_patches) * np.array(patch_size)) / np.array(num_mask_units)

        decoder_class = MAE_Decoder
        if use_crossmae:
            decoder_class = CrossMAE_Decoder

        self.decoder = decoder_class(
            num_patches=num_mask_units,
            spatial_dims=spatial_dims,
            base_patch_size=mask_unit_size.astype(int),
            enc_dim=self.encoder.final_dim,
            emb_dim=decoder_dim,
            num_layer=decoder_layer,
            num_head=decoder_head,
            has_cls_token=False,
        )

    def forward(self, img):
        features, mask, forward_indexes, backward_indexes, save_layers = self.encoder(img)
        features = rearrange(features, "b t c -> t b c")
        predicted_img = self.decoder(features, forward_indexes, backward_indexes)
        return predicted_img, mask
