from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_

from cyto_dl.nn.vits.blocks import CrossAttentionBlock
from cyto_dl.nn.vits.utils import take_indexes


class CrossMAE_Decoder(torch.nn.Module):
    """Decoder inspired by [CrossMAE](https://crossmae.github.io/) where masked tokens only attend
    to visible tokens."""

    def __init__(
        self,
        num_patches: List[int],
        spatial_dims: int = 3,
        base_patch_size: Optional[List[int]] = [4, 8, 8],
        enc_dim: Optional[int] = 768,
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 4,
        num_head: Optional[int] = 3,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int]
            Number of patches in each dimension
        base_patch_size: Tuple[int]
            Size of each patch
        enc_dim: int
            Dimension of encoder
        emb_dim: int
            Dimension of embedding
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        """
        super().__init__()

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
        self.decoder_norm = nn.LayerNorm(emb_dim)
        self.projection_norm = nn.LayerNorm(emb_dim)

        self.projection = torch.nn.Linear(enc_dim, emb_dim)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(np.prod(num_patches) + 1, 1, emb_dim))

        self.head = torch.nn.Linear(emb_dim, torch.prod(torch.as_tensor(base_patch_size)))
        self.num_patches = torch.as_tensor(num_patches)

        if spatial_dims == 3:
            self.patch2img = Rearrange(
                "(n_patch_z n_patch_y n_patch_x) b (c patch_size_z patch_size_y patch_size_x) ->  b c (n_patch_z patch_size_z) (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
                n_patch_z=num_patches[0],
                n_patch_y=num_patches[1],
                n_patch_x=num_patches[2],
                patch_size_z=base_patch_size[0],
                patch_size_y=base_patch_size[1],
                patch_size_x=base_patch_size[2],
            )
        elif spatial_dims == 2:
            self.patch2img = Rearrange(
                "(n_patch_y n_patch_x) b (c patch_size_y patch_size_x) ->  b c (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
                n_patch_y=num_patches[0],
                n_patch_x=num_patches[1],
                patch_size_y=base_patch_size[0],
                patch_size_x=base_patch_size[1],
            )

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, features, forward_indexes, backward_indexes):
        # HACK TODO allow usage of multiple intermediate feature weights, this works when decoder is 0 layers
        features = features.squeeze(0)
        T, B, C = features.shape
        # we could do cross attention between decoder_dim queries and encoder_dim features, but it seems to work fine having both at decoder_dim for now
        features = self.projection_norm(self.projection(features))

        # add cls token
        backward_indexes = torch.cat(
            [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1],
            dim=0,
        )
        forward_indexes = torch.cat(
            [torch.zeros(1, forward_indexes.shape[1]).to(forward_indexes), forward_indexes + 1],
            dim=0,
        )
        # fill in masked regions
        features = torch.cat(
            [
                features,
                self.mask_token.expand(
                    backward_indexes.shape[0] - features.shape[0], features.shape[1], -1
                ),
            ],
            dim=0,
        )

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
            [torch.zeros((T - 1, B, patches.shape[-1]), requires_grad=False).to(patches), patches],
            dim=0,
        )
        patches = take_indexes(patches, backward_indexes[1:] - 1)
        # patches to image
        img = self.patch2img(patches)

        return img
