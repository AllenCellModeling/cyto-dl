# modified from https://github.com/IcarusWizard/MAE/blob/main/model.py#L124

from typing import List, Optional, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

from cyto_dl.nn.vits.blocks import CrossAttentionBlock
from cyto_dl.nn.vits.utils import (
    get_positional_embedding,
    match_tuple_dimensions,
    take_indexes,
)


class MAE_Decoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: Union[int, List[int]],
        spatial_dims: int = 3,
        patch_size: Optional[Union[int, List[int]]] = 4,
        enc_dim: Optional[int] = 768,
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 4,
        num_head: Optional[int] = 3,
        has_cls_token: Optional[bool] = False,
        learnable_pos_embedding: Optional[bool] = True,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int], int
            Number of patches in each dimension. If int, the same number of patches is used for all dimensions.
        patch_size: Tuple[int], int
            Size of each patch. If int, the same patch size is used for all dimensions.
        enc_dim: int
            Dimension of encoder
        emb_dim: int
            Dimension of decoder
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        has_cls_token: bool
            Whether encoder features have a cls token
        learnable_pos_embedding: bool
            If True, learnable positional embeddings are used. If False, fixed sin/cos positional embeddings. Empirically, fixed positional embeddings work better for brightfield images.
        """
        super().__init__()
        num_patches, patch_size = match_tuple_dimensions(spatial_dims, [num_patches, patch_size])

        self.has_cls_token = has_cls_token

        self.projection_norm = nn.LayerNorm(emb_dim)
        self.projection = torch.nn.Linear(enc_dim, emb_dim)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.pos_embedding = get_positional_embedding(
            num_patches, emb_dim, use_cls_token=has_cls_token, learnable=learnable_pos_embedding
        )

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )
        out_dim = torch.prod(torch.as_tensor(patch_size)).item()
        self.decoder_norm = nn.LayerNorm(emb_dim)
        self.head = torch.nn.Linear(emb_dim, out_dim)
        self.num_patches = torch.as_tensor(num_patches)

        if spatial_dims == 3:
            self.patch2img = Rearrange(
                "(n_patch_z n_patch_y n_patch_x) b (c patch_size_z patch_size_y patch_size_x) ->  b c (n_patch_z patch_size_z) (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
                n_patch_z=num_patches[0],
                n_patch_y=num_patches[1],
                n_patch_x=num_patches[2],
                patch_size_z=patch_size[0],
                patch_size_y=patch_size[1],
                patch_size_x=patch_size[2],
            )
        elif spatial_dims == 2:
            self.patch2img = Rearrange(
                "(n_patch_y n_patch_x) b (c patch_size_y patch_size_x) ->  b c (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
                n_patch_y=num_patches[0],
                n_patch_x=num_patches[1],
                patch_size_y=patch_size[0],
                patch_size_x=patch_size[1],
            )

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)

    def adjust_indices_for_cls(self, indexes):
        if self.has_cls_token:
            # add all zeros to indices - this keeps the class token as the first index in the
            # tensor. We also have to add 1 to all the indices to account for the zeros we added
            return torch.cat(
                [
                    torch.zeros(1, indexes.shape[1], device=indexes.device, dtype=torch.long),
                    indexes + 1,
                ],
                dim=0,
            )
        return indexes

    def add_mask_tokens(self, features, backward_indexes):
        # fill in deleted masked regions with mask token
        num_visible_tokens, B, _ = features.shape
        # total number of tokens - number of visible tokens
        num_mask_tokens = backward_indexes.shape[0] - num_visible_tokens
        mask_tokens = repeat(self.mask_token, "1 1 c -> t b c", t=num_mask_tokens, b=B)
        return torch.cat([features, mask_tokens], dim=0)

    def forward(self, features, forward_indexes, backward_indexes):
        # project from encoder dimension to decoder dimension
        features = self.projection_norm(self.projection(features))

        backward_indexes = self.adjust_indices_for_cls(backward_indexes)

        features = self.add_mask_tokens(features, backward_indexes)

        # unshuffle to original positions
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        # decode
        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")

        if self.has_cls_token:
            features = features[1:]  # remove global feature

        # (npatches x npatches x npatches) b (emb dim) -> (npatches* npatches * npatches) b (z y x)
        patches = self.head(self.decoder_norm(features))

        # patches to image
        img = self.patch2img(patches)
        return img


class CrossMAE_Decoder(MAE_Decoder):
    """Decoder inspired by [CrossMAE](https://crossmae.github.io/) where masked tokens only attend
    to visible tokens."""

    def __init__(
        self,
        num_patches: Union[int, List[int]],
        spatial_dims: int = 3,
        patch_size: Optional[Union[int, List[int]]] = 4,
        enc_dim: Optional[int] = 768,
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 4,
        num_head: Optional[int] = 3,
        has_cls_token: Optional[bool] = True,
        learnable_pos_embedding: Optional[bool] = True,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int], int
            Number of patches in each dimension. If int, the same number of patches is used for all dimensions.
        patch_size: Tuple[int]
            Size of each patch in each dimension. If int, the same patch size is used for all dimensions.
        enc_dim: int
            Dimension of encoder
        emb_dim: int
            Dimension of embedding
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        has_cls_token: bool
            Whether encoder features have a cls token
        learnable_pos_embedding: bool
            If True, learnable positional embeddings are used. If False, fixed sin/cos positional embeddings are used. Empirically, fixed positional embeddings work better for brightfield images.
        """
        super().__init__(
            num_patches,
            spatial_dims,
            patch_size,
            enc_dim,
            emb_dim,
            num_layer,
            num_head,
            has_cls_token,
            learnable_pos_embedding,
        )

        self.transformer = torch.nn.ParameterList(
            [
                CrossAttentionBlock(
                    encoder_dim=emb_dim,
                    decoder_dim=emb_dim,
                    num_heads=num_head,
                )
                for _ in range(num_layer)
            ]
        )

    def forward(self, features, forward_indexes, backward_indexes):
        # HACK TODO allow usage of multiple intermediate feature weights, this works when decoder is 1 layer
        # features can be n t b c (if intermediate feature weighter used) or t b c if not
        features = features[0] if len(features.shape) == 4 else features
        T, B, C = features.shape
        # we could do cross attention between decoder_dim queries and encoder_dim features, but it seems to work fine having both at decoder_dim for now
        features = self.projection_norm(self.projection(features))

        backward_indexes = self.adjust_indices_for_cls(backward_indexes)
        forward_indexes = self.adjust_indices_for_cls(forward_indexes)

        features = self.add_mask_tokens(features, backward_indexes)

        # unshuffle to original positions for positional embedding so we can do cross attention during decoding
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        # reshuffle to shuffled positions for cross attention
        features = take_indexes(features, forward_indexes)
        features, masked = features[:T], features[T:]

        masked = rearrange(masked, "t b c -> b t c")
        features = rearrange(features, "t b c -> b t c")

        for transformer in self.transformer:
            masked = transformer(masked, features)

        # noneed to remove cls token, it's a part of the features vector
        masked = rearrange(masked, "b t c -> t b c")

        # (npatches x npatches x npatches) b (emb dim) -> (npatches* npatches * npatches) b (z y x)
        masked = self.decoder_norm(masked)
        patches = self.head(masked)

        # add back in visible/encoded tokens that we don't calculate loss on
        patches = torch.cat(
            [
                torch.zeros(
                    # T-1 accounts for cls token
                    (T - self.has_cls_token, B, patches.shape[-1]),
                    requires_grad=False,
                    device=patches.device,
                    dtype=patches.dtype,
                ),
                patches,
            ],
            dim=0,
        )
        patches = (
            take_indexes(patches, backward_indexes[1:] - 1)
            if self.has_cls_token
            else take_indexes(patches, backward_indexes)
        )
        # patches to image
        img = self.patch2img(patches)
        return img
