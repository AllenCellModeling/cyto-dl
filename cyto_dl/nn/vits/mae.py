# modified from https://github.com/IcarusWizard/MAE/blob/main/model.py#L124

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

from cyto_dl.nn.vits.cross_mae import CrossMAE_Decoder


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(
        sequences, 0, repeat(indexes.to(sequences.device), "t b -> t b c", c=sequences.shape[-1])
    )


class Patchify(torch.nn.Module):
    """Class for converting images to a masked sequence of patches with positional embeddings."""

    def __init__(
        self,
        patch_size: List[int],
        emb_dim: int,
        n_patches: List[int],
        spatial_dims: int = 3,
        context_pixels: List[int] = [0, 0, 0],
        input_channels: int = 1,
    ):
        """
        Parameters
        ----------
        patch_size: List[int]
            Size of each patch
        emb_dim: int
            Dimension of encoder
        n_patches: List[int]
            Number of patches in each spatial dimension
        spatial_dims: int
            Number of spatial dimensions
        context_pixels: List[int]
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension.
        input_channels: int
            Number of input channels
        """
        super().__init__()
        self.n_patches = np.asarray(n_patches)
        self.spatial_dims = spatial_dims

        self.pos_embedding = torch.nn.Parameter(torch.zeros(np.prod(n_patches), 1, emb_dim))

        context_pixels = context_pixels[:spatial_dims]
        weight_size = np.asarray(patch_size) + np.round(np.array(context_pixels) * 2).astype(int)

        if spatial_dims == 3:
            self.conv = nn.Conv3d(
                in_channels=input_channels,
                out_channels=emb_dim,
                kernel_size=weight_size,
                stride=patch_size,
                padding=context_pixels,
            )
            self.img2token = Rearrange("b c z y x -> (z y x) b c")
            self.patch2img = Rearrange(
                "(n_patch_z n_patch_y n_patch_x) b c ->  b c n_patch_z n_patch_y n_patch_x",
                n_patch_z=n_patches[0],
                n_patch_y=n_patches[1],
                n_patch_x=n_patches[2],
            )
        elif spatial_dims == 2:
            self.conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=emb_dim,
                kernel_size=weight_size,
                stride=patch_size,
                padding=context_pixels,
            )
            self.img2token = Rearrange("b c y x -> (y x) b c")
            self.patch2img = Rearrange(
                "(n_patch_y n_patch_x) b c ->  b c n_patch_y n_patch_x",
                n_patch_y=n_patches[0],
                n_patch_x=n_patches[1],
            )

        self._init_weight()

    def _init_weight(self):
        trunc_normal_(self.pos_embedding, std=0.02)

    def get_mask(self, img, n_visible_patches, num_patches):
        B = img.shape[0]

        indexes = [random_indexes(num_patches) for _ in range(B)]
        # forward indexes : index in image -> shuffledpatch
        forward_indexes = torch.as_tensor(
            np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long
        )
        # backward indexes : shuffled patch -> index in image
        backward_indexes = torch.as_tensor(
            np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long
        )

        mask = torch.zeros(num_patches, B, 1)
        # visible patches are first
        mask[:n_visible_patches] = 1
        mask = take_indexes(mask, backward_indexes)
        mask = self.patch2img(mask)
        # one pixel per masked patch, interpolate to size of input image
        mask = torch.nn.functional.interpolate(
            mask, img.shape[-self.spatial_dims :], mode="nearest"
        )

        return mask.to(img), forward_indexes, backward_indexes

    def forward(self, img, mask_ratio):
        # generate mask
        num_patches = np.prod(self.n_patches)
        n_visible_patches = int(num_patches * (1 - mask_ratio))
        mask, forward_indexes, backward_indexes = self.get_mask(
            img, n_visible_patches, num_patches
        )
        # generate patches
        tokens = self.conv(img * mask)
        tokens = self.img2token(tokens)
        # add position embedding
        tokens = tokens + self.pos_embedding
        if mask_ratio > 0:
            # extract visible patches
            tokens = take_indexes(tokens, forward_indexes)[:n_visible_patches]

        # mask is used above to mask out patches, we need to invert it for loss calculation
        mask = (1 - mask).bool()

        return tokens, mask, forward_indexes, backward_indexes


class MAE_Encoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: List[int],
        spatial_dims: int = 3,
        base_patch_size: List[int] = (16, 16, 16),
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 12,
        num_head: Optional[int] = 3,
        context_pixels: Optional[List[int]] = [0, 0, 0],
        input_channels: Optional[int] = 1,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int]
            Number of patches in each dimension
        spatial_dims: int
            Number of spatial dimensions
        base_patch_size: List[int]
            Size of each patch
        emb_dim: int
            Dimension of embedding
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        context_pixels: List[int]
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension.
        input_channels: int
            Number of input channels
        """
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.patchify = Patchify(
            base_patch_size, emb_dim, num_patches, spatial_dims, context_pixels, input_channels
        )

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, img, mask_ratio=0.75):
        patches, mask, forward_indexes, backward_indexes = self.patchify(img, mask_ratio)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")
        if mask_ratio > 0:
            return features, mask, forward_indexes, backward_indexes
        return features


class MAE_Decoder(torch.nn.Module):
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
            Dimension of decoder
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        """
        super().__init__()
        self.projection_norm = nn.LayerNorm(emb_dim)
        self.projection = torch.nn.Linear(enc_dim, emb_dim)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(np.prod(num_patches) + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )
        out_dim = torch.prod(torch.as_tensor(base_patch_size)).item()
        self.head_norm = nn.LayerNorm(out_dim)
        self.head = torch.nn.Linear(emb_dim, out_dim)
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
        # project from encoder dimension to decoder dimension
        features = self.projection_norm(self.projection(features))

        backward_indexes = torch.cat(
            [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1],
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
        # unshuffle to original positions
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        # decode
        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature

        # (npatches x npatches x npatches) b (emb dim) -> (npatches* npatches * npatches) b (z y x)
        patches = self.head_norm(self.head(features))

        # patches to image
        img = self.patch2img(patches)
        return img


class MAE_ViT(torch.nn.Module):
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
        decoder_dim: Optional[int] = 192,
        mask_ratio: Optional[int] = 0.75,
        use_crossmae: Optional[bool] = False,
        context_pixels: Optional[List[int]] = [0, 0, 0],
        input_channels: Optional[int] = 1,
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
        use_crossmae: bool
            Use CrossMAE-style decoder
        context_pixels: List[int]
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension.
        input_channels: int
        """
        super().__init__()
        assert spatial_dims in (2, 3), "Spatial dims must be 2 or 3"

        if isinstance(num_patches, int):
            num_patches = [num_patches] * spatial_dims
        if isinstance(base_patch_size, int):
            base_patch_size = [base_patch_size] * spatial_dims

        assert len(num_patches) == spatial_dims, "num_patches must be of length spatial_dims"
        assert (
            len(base_patch_size) == spatial_dims
        ), "base_patch_size must be of length spatial_dims"

        self.mask_ratio = mask_ratio

        self.encoder = MAE_Encoder(
            num_patches,
            spatial_dims,
            base_patch_size,
            emb_dim,
            encoder_layer,
            encoder_head,
            context_pixels,
            input_channels,
        )

        decoder_class = MAE_Decoder
        if use_crossmae:
            decoder_class = CrossMAE_Decoder
        self.decoder = decoder_class(
            num_patches=num_patches,
            spatial_dims=spatial_dims,
            base_patch_size=base_patch_size,
            enc_dim=emb_dim,
            emb_dim=decoder_dim,
            num_layer=decoder_layer,
            num_head=decoder_head,
        )

    def forward(self, img):
        features, mask, forward_indexes, backward_indexes = self.encoder(img, self.mask_ratio)
        predicted_img = self.decoder(features, forward_indexes, backward_indexes)
        return predicted_img, mask
