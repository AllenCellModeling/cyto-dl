# modified from https://github.com/IcarusWizard/MAE/blob/main/model.py#L124

import numpy as np
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from monai.networks.blocks import UnetBasicBlock, UnetOutBlock
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
    def __init__(self, base_patch_size, emb_dim, n_patches):
        super().__init__()
        self.n_patches = n_patches
        self.weight = torch.nn.Parameter(torch.zeros(emb_dim, 1, *base_patch_size))
        self.norm = torch.nn.LayerNorm([emb_dim, n_patches, n_patches, n_patches])
        self.emb_dim = emb_dim

    def resample_weight(self, length):
        return torch.nn.functional.interpolate(self.weight, size=length, mode="trilinear")

    def forward(self, img):
        # all images in batch assumed to be same resolution
        patch_size = [int(s / self.n_patches) for s in img.shape[-3:]]
        assert np.all(
            np.asarray(patch_size) > 1
        ), "All patch size must be >1 in all dimensions, got {patches}. Increase physical_crop_size or decrease number of patches. "
        tokens = torch.nn.functional.conv3d(
            img, weight=self.resample_weight(patch_size), stride=patch_size
        )
        tokens = self.norm(tokens)
        return tokens, patch_size


class MAE_Encoder(torch.nn.Module):
    def __init__(
        self,
        num_patches,
        base_patch_size=(4, 8, 8),
        emb_dim=192,
        num_layer=12,
        num_head=3,
        mask_ratio=0.75,
    ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(num_patches**3, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = Patchify(base_patch_size, emb_dim, num_patches)

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, img):
        patches, patch_size = self.patchify(img)
        patches = rearrange(patches, "b c z y x -> (z y x) b c")
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")

        return features, backward_indexes, patch_size


class MAE_Decoder(torch.nn.Module):
    def __init__(
        self,
        num_patches,
        base_patch_size=(4, 8, 8),
        emb_dim=192,
        num_layer=4,
        num_head=3,
    ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(num_patches**3 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.head = torch.nn.Linear(emb_dim, torch.prod(torch.as_tensor(base_patch_size)))
        self.num_patches = num_patches

        # n_conv_layers= 16
        # self.conv = torch.nn.Sequential(UnetBasicBlock(spatial_dims = 3, in_channels = 1, out_channels = n_conv_layers, kernel_size = 3, stride = 1, norm_name = 'instance'),
        # UnetOutBlock(spatial_dims =3, in_channels = n_conv_layers, out_channels = 1))

        self.patch2img = Rearrange(
            "(n_patch_z n_patch_y n_patch_x) b (c patch_size_z patch_size_y patch_size_x) ->  b c (n_patch_z patch_size_z) (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
            n_patch_z=num_patches,
            n_patch_y=num_patches,
            n_patch_x=num_patches,
            patch_size_z=base_patch_size[0],
            patch_size_y=base_patch_size[1],
            patch_size_x=base_patch_size[2],
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
            img, tuple(torch.as_tensor(patch_size) * self.num_patches), mode="trilinear"
        )
        # img = self.conv(img)

        mask = self.patch2img(mask)
        mask = torch.nn.functional.interpolate(
            mask, tuple(torch.as_tensor(patch_size) * self.num_patches), mode="nearest"
        )

        return img, mask


class MAE_ViT(torch.nn.Module):
    def __init__(
        self,
        num_patches=8,
        base_patch_size=[10, 20, 20],
        emb_dim=192,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        mask_ratio=0.75,
    ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(
            num_patches, base_patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio
        )
        self.decoder = MAE_Decoder(
            num_patches, base_patch_size, emb_dim, decoder_layer, decoder_head
        )

    def forward(self, img):
        features, backward_indexes, patch_size = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes, patch_size)
        return predicted_img, mask
