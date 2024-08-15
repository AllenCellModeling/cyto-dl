from typing import List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_

from cyto_dl.nn.vits.mae import MAE_Decoder
from cyto_dl.nn.vits.blocks import CrossAttentionBlock
from cyto_dl.nn.vits.utils import get_positional_embedding, take_indexes


class CrossMAE_Decoder(MAE_decoder):
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
        has_cls_token: Optional[bool] = True,
        learnable_pos_embedding: Optional[bool] = True,
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
        has_cls_token: bool
            Whether encoder features have a cls token
        learnable_pos_embedding: bool
            If True, learnable positional embeddings are used. If False, fixed sin/cos positional embeddings are used. Empirically, fixed positional embeddings work better for brightfield images.
        """
        super().__init__(num_patches, spatial_dims, base_patch_size, enc_dim, emb_dim, num_layer, num_head, has_cls_token, learnable_pos_embedding)

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
        # HACK TODO allow usage of multiple intermediate feature weights, this works when decoder is 0 layers
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
                    (T - 1, B, patches.shape[-1]),
                    requires_grad=False,
                    device=patches.device,
                    dtype=patches.dtype,
                ),
                patches,
            ],
            dim=0,
        )
        patches = take_indexes(patches, backward_indexes[1:] - 1) if self.has_cls_token else take_indexes(patches, backward_indexes)
        # patches to image
        img = self.patch2img(patches)
        return img
