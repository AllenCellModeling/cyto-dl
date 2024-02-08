# modified from https://github.com/IcarusWizard/MAE/blob/main/model.py#L124

from typing import List, Optional

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1]))


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(
            np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)
        backward_indexes = torch.as_tensor(
            np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


class Patchify(torch.nn.Module):
    # based on https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py
    """Class for flexibly turning image into sequence of patches.

    Convolutional weights are resized to match the `base_patch_size`.
    """

    def __init__(self, base_patch_size, emb_dim, n_patches, spatial_dims=3):
        super().__init__()
        self.n_patches = np.asarray(n_patches)
        self.weight = torch.nn.Parameter(torch.zeros(emb_dim, 1, *base_patch_size))
        self.norm = torch.nn.LayerNorm([emb_dim, *n_patches[:spatial_dims]])
        self.emb_dim = emb_dim
        self.spatial_dims = spatial_dims
        self.conv = torch.nn.functional.conv3d if spatial_dims == 3 else torch.nn.functional.conv2d

    def resample_weight(self, length):
        return torch.nn.functional.interpolate(self.weight, size=length)

    def forward(self, img):
        patch_size = (
            (np.asarray(img.shape[-self.spatial_dims :]) / self.n_patches).astype(int).tolist()
        )
        tokens = self.conv(img, weight=self.resample_weight(patch_size), stride=patch_size)
        tokens = self.norm(tokens)
        assert np.all(tokens.shape[-self.spatial_dims :] == self.n_patches)
        return tokens, patch_size


class MAE_Encoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: List[int],
        spatial_dims: int = 3,
        base_patch_size: List[int] = (16, 16, 16),
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 12,
        num_head: Optional[int] = 3,
        mask_ratio: Optional[int] = 0.75,
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
        mask_ratio: float
            Ratio of patches to mask out
        """
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(np.prod(num_patches), 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.patchify = Patchify(base_patch_size, emb_dim, num_patches, spatial_dims)

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        if spatial_dims == 3:
            self.img2token = Rearrange("b c z y x -> (z y x) b c")
        elif spatial_dims == 2:
            self.img2token = Rearrange("b c y x -> (y x) b c")

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, img, do_mask=True):
        patches, patch_size = self.patchify(img)
        patches = self.img2token(patches)
        patches = patches + self.pos_embedding

        backward_indexes = None
        if do_mask:
            patches, _, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")
        if do_mask:
            return features, backward_indexes, patch_size
        return features


class MAE_Decoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: List[int],
        spatial_dims: int = 3,
        base_patch_size: Optional[List[int]] = [4, 8, 8],
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
        emb_dim: int
            Dimension of embedding
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        """
        super().__init__()
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(np.prod(num_patches) + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

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

    def forward(self, features, backward_indexes, patch_size):
        T = features.shape[0]
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
        patches = self.head(features)
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
        mask_ratio: Optional[int] = 0.75,
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

        self.encoder = MAE_Encoder(
            num_patches,
            spatial_dims,
            base_patch_size,
            emb_dim,
            encoder_layer,
            encoder_head,
            mask_ratio,
        )
        self.decoder = MAE_Decoder(
            num_patches, spatial_dims, base_patch_size, emb_dim, decoder_layer, decoder_head
        )

    def forward(self, img):
        features, backward_indexes, patch_size = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes, patch_size)
        return predicted_img, mask
