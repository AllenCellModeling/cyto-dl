# modified from https://github.com/IcarusWizard/MAE/blob/main/model.py#L124

from typing import List, Optional

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_

from cyto_dl.nn.vits.blocks import CrossSelfBlock
from cyto_dl.nn.vits.mae import MAE_Encoder, take_indexes


class SiamMAE_Decoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: List[int],
        spatial_dims: int = 3,
        base_patch_size: Optional[List[int]] = [4, 8, 8],
        enc_dim: Optional[int] = 768,
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 4,
        num_head: Optional[int] = 4,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int]
            Number of patches in each dimension
        base_patch_size: Tuple[int]
            Size of each patch
        emb_dim: int
            Dimension of embedding
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        """
        super().__init__()
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        # add 1 for cls token
        self.pos_embedding = torch.nn.Parameter(torch.zeros(np.prod(num_patches) + 1, 1, emb_dim))
        self.transformer = torch.nn.ModuleList(
            [CrossSelfBlock(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.t0_projection_norm = torch.nn.LayerNorm(emb_dim)
        self.t0_projection = torch.nn.Linear(enc_dim, emb_dim)

        self.t1_projection_norm = torch.nn.LayerNorm(emb_dim)
        self.t1_projection = torch.nn.Linear(enc_dim, emb_dim)

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

    def forward(self, features_t0, features_t1, backward_indexes, patch_size):
        T = features_t1.shape[0]

        # project to encoder dimension
        features_t0 = self.t0_projection_norm(self.t0_projection(features_t0))
        features_t1 = self.t1_projection_norm(self.t1_projection(features_t1))

        # backward index is index to unshuffle that position to - cls token is 0 so we add an all-0 tensor to the front
        backward_indexes = torch.cat(
            [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1],
            dim=0,
        )
        # fill in masked regions in t1
        features_t1 = torch.cat(
            [
                features_t1,
                self.mask_token.expand(
                    backward_indexes.shape[0] - features_t1.shape[0], features_t1.shape[1], -1
                ),
            ],
            dim=0,
        )
        # unshuffle t1 features to original positions
        features_t1 = take_indexes(features_t1, backward_indexes)
        features_t1 = features_t1 + self.pos_embedding
        features_t1 = rearrange(features_t1, "t b c -> b t c")

        # add pos emb to t0
        features_t0 = features_t0 + self.pos_embedding
        features_t0 = rearrange(features_t0, "t b c -> b t c")

        # cross attention decoder
        for transformer in self.transformer:
            features_t1 = transformer(features_t1, features_t0)

        # returned features are only used to reconstruct t1
        features_t1 = rearrange(features_t1, "b t c -> t b c")
        features_t1 = features_t1[1:]  # remove global feature

        # (npatches x npatches x npatches) b (emb dim) -> (npatches* npatches * npatches) b (z y x)
        patches = self.head(features_t1)
        mask = torch.zeros_like(patches)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        # patches to image
        img = self.patch2img(patches)
        img = torch.nn.functional.interpolate(
            img, tuple(torch.as_tensor(patch_size) * self.num_patches)
        )

        mask = self.patch2img(mask)
        mask = torch.nn.functional.interpolate(
            mask, tuple(torch.as_tensor(patch_size) * self.num_patches), mode="nearest"
        )
        return img, mask


class SiamMAE(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        num_patches: Optional[List[int]] = [2, 32, 32],
        base_patch_size: Optional[List[int]] = [16, 16, 16],
        emb_dim: Optional[int] = 768,
        encoder_layer: Optional[int] = 12,
        encoder_head: Optional[int] = 8,
        decoder_layer: Optional[int] = 4,
        decoder_head: Optional[int] = 8,
        decoder_dim: Optional[int] = 128,
    ) -> None:
        """
        Parameters
        ----------
        spatial_dims: int
            Number of spatial dimensions
        num_patches: List[int]
            Number of patches in each dimension (ZYX order)
        base_patch_size: List[int]
            Size of each patch (ZYX order)
        emb_dim: int
            Dimension of encoder embedding
        encoder_layer: int
            Number of encoder transformer layers
        encoder_head: int
            Number of encoder heads
        decoder_layer: int
            Number of decoder transformer layers
        decoder_head: int
            Number of decoder heads
        mask_ratio: float
            Ratio of patches to mask out
        """
        assert spatial_dims in (2, 3), "Spatial dims must be 2 or 3"
        assert len(num_patches) == spatial_dims, "num_patches must be of length spatial_dims"
        assert (
            len(base_patch_size) == spatial_dims
        ), "base_patch_size must be of length spatial_dims"

        super().__init__()

        if isinstance(num_patches, int):
            num_patches = [num_patches] * 3
        if isinstance(base_patch_size, int):
            base_patch_size = [base_patch_size] * 3

        self.encoder = MAE_Encoder(
            num_patches, spatial_dims, base_patch_size, emb_dim, encoder_layer, encoder_head
        )
        self.decoder = SiamMAE_Decoder(
            num_patches,
            spatial_dims,
            base_patch_size,
            emb_dim,
            decoder_dim,
            decoder_layer,
            decoder_head,
        )

    def forward(self, img):
        # first channel is t0, second channel is t1
        t0, t1 = img[:, :1], img[:, 1:]
        features_t0 = self.encoder(t0, mask_ratio=0)
        features_t1, forward_indexes, backward_indexes, patch_size = self.encoder(
            t1, mask_ratio=0.95
        )
        predicted_img, mask = self.decoder(features_t0, features_t1, backward_indexes, patch_size)
        return predicted_img, mask


# class SandwichSiamMAE(torch.nn.Module):
#     def __init__(self, meat_mask_ratio,
#         spatial_dims: int = 3,
#         num_patches: Optional[List[int]] = [2, 32, 32],
#         base_patch_size: Optional[List[int]] = [16, 16, 16],
#         emb_dim: Optional[int] = 768,
#         encoder_layer: Optional[int] = 12,
#         encoder_head: Optional[int] = 8,
#         decoder_layer: Optional[int] = 4,
#         decoder_head: Optional[int] = 8,
#         decoder_dim: Optional[int] = 128,
#         mask_ratio: Optional[int] = 0.75,
#     ):
#         assert spatial_dims in [2, 3], "Spatial dims must be 2 or 3"
#         assert len(num_patches) == spatial_dims, "num_patches must be of length spatial_dims"
#         assert len(base_patch_size) == spatial_dims, "base_patch_size must be of length spatial_dims"

#         super().__init__()

#         if isinstance(num_patches, int):
#             num_patches = [num_patches] * 3
#         if isinstance(base_patch_size, int):
#             base_patch_size = [base_patch_size] * 3

#         self.encoder = MAE_Encoder(
#             num_patches, spatial_dims, base_patch_size, emb_dim, encoder_layer, encoder_head
#         )
#         self.meat_mask_ratio = meat_mask_ratio
#         self.mask_ratio =mask_ratio
#         self.decoder = SandwichDecoder(num_patches, spatial_dims, base_patch_size, emb_dim, decoder_dim, decoder_layer, decoder_head)

#     def forward(self, img):
#         features_prev,  backward_indexes_prev, _ = self.encoder(img[:, :1], self.mask_ratio)
#         features_next,  backward_indexes_next, _ = self.encoder(img[:, 2:], self.mask_ratio)

#         features_current, backward_indexes, patch_size = self.encoder(img[:, 1:2], self.meat_mask_ratio)

#         data = {
#             "prev": (features_prev, backward_indexes_prev),
#             "next": (features_next, backward_indexes_next),
#             "current": (features_current, backward_indexes),
#         }

#         predicted_img, mask = self.decoder(data, patch_size)
#         return predicted_img, mask

from timm.models.vision_transformer import Block

from cyto_dl.nn.vits.blocks.cross_attention import CrossAttentionBlock


class SandwichBlock(torch.nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attention = CrossAttentionBlock(
            encoder_dim=dim, decoder_dim=dim, num_heads=num_heads
        )
        self.self_attention = Block(dim, num_heads)

    def forward(self, prev, current, next):
        current = self.cross_attention(current, prev)
        current = self.cross_attention(current, next)
        current = self.self_attention(current)
        return current


# class SandwichDecoder(SiamMAE_Decoder):
#     def __init__(
#         self,
#         num_patches: List[int],
#         spatial_dims: int = 3,
#         base_patch_size: Optional[List[int]] = [4, 8, 8],
#         enc_dim: Optional[int] = 512,
#         emb_dim: Optional[int] = 192,
#         num_layer: Optional[int] = 4,
#         num_head: Optional[int] = 3,
#     ) -> None:
#         super().__init__(num_patches, spatial_dims, base_patch_size, emb_dim, num_layer, num_head)
#         # distinguish between previous and next frame
#         self.temporal_embedding = torch.nn.ParameterDict({
#             'prev': torch.nn.Parameter(torch.zeros(1, 1, emb_dim)),
#             'next': torch.nn.Parameter(torch.zeros(1, 1, emb_dim)),
#         })
#         self.transformer = torch.nn.ModuleList([SandwichBlock(emb_dim, num_head) for _ in range(num_layer)])
#         self.project = torch.nn.Linear(enc_dim, emb_dim)

#     def add_pos_enc_to_features(self, data):
#         features, backward_indexes = data
#         # account for cls token by adding all-0 tensor to front of backward indexes
#         backward_indexes = torch.cat(
#             [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1],
#             dim=0,
#         )
#         # fill in masked regions in t1
#         features = torch.cat(
#             [
#                 features,
#                 self.mask_token.expand(
#                     backward_indexes.shape[0] - features.shape[0], features.shape[1], -1
#                 ),
#             ],
#             dim=0,
#         )
#         # unshuffle t1 features to original positions
#         features = take_indexes(features, backward_indexes)
#         features = features + self.pos_embedding
#         return features, backward_indexes

#     def add_sparse_positional_embedding(self, data):
#         features, backward_indexes = data
#         T, B, C= features.shape
#         # account for cls token by adding all-0 tensor to front of backward indexes
#         backward_indexes = torch.cat(
#             [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1],
#             dim=0,
#         )
#         forward_indexes = torch.argsort(backward_indexes, dim=0)
#         sparse_pe = take_indexes(self.pos_embedding.repeat(1, B, 1), forward_indexes)
#         sparse_pe = sparse_pe[:T]
#         features = features + sparse_pe
#         return features

#     def project_to_decode_dim(self, data):
#         features, backward_indexes = data
#         features = self.project(features)
#         return features, backward_indexes


#     def forward(self, data, patch_size):
#         #todo add linear layer for features_t1?
#         T = data['prev'][0].shape[0]

#         data = {k: self.project_to_decode_dim(v) for k, v in data.items()}

#         features_prev= self.add_sparse_positional_embedding(data['prev'])
#         features_prev = features_prev + self.temporal_embedding['prev'].expand(features_prev.shape[0], -1, -1)
#         features_prev = rearrange(features_prev, "t b c -> b t c")

#         features_next= self.add_sparse_positional_embedding(data['next'])
#         features_next = features_next + self.temporal_embedding['next'].expand(features_next.shape[0], -1, -1)
#         features_next = rearrange(features_next, "t b c -> b t c")

#         features_current, backward_indexes = self.add_pos_enc_to_features(data['current'])
#         features_current = rearrange(features_current, "t b c -> b t c")

#         # cross attention decoder
#         for transformer in self.transformer:
#             features_current = transformer(features_prev, features_current, features_next)

#         # returned features are only used to reconstruct t1
#         features_current = rearrange(features_current, "b t c -> t b c")
#         features_current = features_current[1:]  # remove global feature

#         # (npatches x npatches x npatches) b (emb dim) -> (npatches* npatches * npatches) b (z y x)
#         patches = self.head(features_current)
#         mask = torch.zeros_like(patches)
#         mask[T:] = 1
#         mask = take_indexes(mask, backward_indexes[1:] - 1)
#         # patches to image
#         img = self.patch2img(patches)
#         img = torch.nn.functional.interpolate(
#             img, tuple(torch.as_tensor(patch_size) * self.num_patches)
#         )

#         mask = self.patch2img(mask)
#         mask = torch.nn.functional.interpolate(
#             mask, tuple(torch.as_tensor(patch_size) * self.num_patches), mode="nearest"
#         )
#         return img, mask
