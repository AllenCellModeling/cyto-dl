from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

from cyto_dl.nn.vits.blocks import CrossAttentionBlock, Patchify
from cyto_dl.nn.vits.utils import take_indexes


class JEPAEncoder(torch.nn.Module):
    def __init__(
        self,
        num_patches: List[int],
        spatial_dims: int = 3,
        patch_size: List[int] = (16, 16, 16),
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
        patch_size: List[int]
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
        self.patchify = Patchify(
            patch_size, emb_dim, num_patches, spatial_dims, context_pixels, input_channels
        )

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

    def transformer_forward(self, patches):
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        return features

    def forward(self, img):
        patches, _, _, _ = self.patchify(img, mask_ratio=0)
        return self.transformer_forward(patches)


class JEPAPredictor(torch.nn.Module):
    def __init__(
        self,
        num_patches: List[int],
        input_dim: Optional[int] = 192,
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 12,
        num_head: Optional[int] = 3,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int]
            Number of patches in each dimension
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

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(np.prod(num_patches), 1, emb_dim))

        self.predictor_embed = torch.nn.Linear(input_dim, emb_dim)

        self.projector_embed = torch.nn.Linear(emb_dim, input_dim)
        self.norm = torch.nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, context_emb, target_masks):
        t, b = target_masks.shape
        # map context embedding to predictor dimension
        context_emb = self.predictor_embed(context_emb)

        # add masked positional embedding to mask tokens
        mask = self.mask_token.expand(t, b, -1)
        pe = self.pos_embedding.expand(-1, b, -1)
        pe = take_indexes(pe, target_masks)
        mask = mask + pe
        mask = rearrange(mask, "t b c -> b t c")
        # cross attention from mask tokens to context embedding
        for transformer in self.transformer:
            mask = transformer(mask, context_emb)
        # norm and project back to input dimension
        mask = self.projector_embed(self.norm(mask))
        return mask


from monai.networks.blocks import UnetOutBlock, UnetResBlock


class JEPASeg(torch.nn.Module):
    """Class for training a simple convolutional decoder on top of a pretrained ViT backbone."""

    def __init__(
        self,
        spatial_dims: int = 3,
        num_patches: Optional[List[int]] = [2, 32, 32],
        patch_size: Optional[List[int]] = [16, 16, 16],
        emb_dim: Optional[int] = 768,
        encoder_ckpt: Optional[str] = None,
        freeze_encoder: Optional[bool] = True,
        **encoder_kwargs,
    ) -> None:
        """
        Parameters
        ----------
        spatial_dims: Optional[int]=3
            Number of spatial dimensions
        num_patches: Optional[List[int]]=[2, 32, 32]
            Number of patches in each dimension (ZYX) order
        patch_size: Optional[List[int]]=[16, 16, 16]
            Base patch size in each dimension (ZYX) order
        emb_dim: Optional[int] =768
            Embedding dimension of ViT backbone
        encoder_layer: Optional[int] =12
            Number of layers in ViT backbone
        encoder_head: Optional[int] =8
            Number of heads in ViT backbone
        decoder_layer: Optional[int] =3
            Number of layers in convolutional decoder
        n_decoder_filters: Optional[int] =16
            Number of filters in convolutional decoder
        out_channels: Optional[int] =6
            Number of output channels in convolutional decoder. Should be 6 for instance segmentation.
        mask_ratio: Optional[int] =0.75
            Ratio of patches to be masked out during training
        upsample_factor:Optional[List[int]] = [2.6134, 2.5005, 2.5005]
            Upsampling factor for each dimension (ZYX) order. Default is AICS 20x to 100x object upsampling
        encoder_ckpt: Optional[str]=None
            Path to pretrained ViT backbone checkpoint
        """
        super().__init__()
        assert spatial_dims in (2, 3)
        if isinstance(num_patches, int):
            num_patches = [num_patches] * spatial_dims
        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial_dims
        assert len(num_patches) == spatial_dims
        assert len(patch_size) == spatial_dims

        self.encoder = JEPAEncoder(
            spatial_dims=spatial_dims,
            num_patches=num_patches,
            patch_size=patch_size,
            emb_dim=emb_dim,
            **encoder_kwargs,
        )
        if encoder_ckpt is not None:
            model = torch.load(encoder_ckpt, map_location="cuda:0")
            enc_state_dict = {
                k.replace("backbone.encoder.", ""): v
                for k, v in model["state_dict"].items()
                if "encoder" in k and "intermediate" not in k
            }
            self.encoder.load_state_dict(enc_state_dict, strict=False)

        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                # allow different weighting of internal activations for finetuning
                param.requires_grad = "intermediate_weighter" in name

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 16 * emb_dim),
            Rearrange(
                "b (n_patch_z n_patch_y n_patch_x) c -> b c n_patch_z n_patch_y n_patch_x",
                n_patch_z=num_patches[0],
                n_patch_y=num_patches[1],
                n_patch_x=num_patches[2],
            ),
            UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=4096,
                out_channels=256,
                stride=1,
                kernel_size=3,
                norm_name="INSTANCE",
                dropout=0,
            ),
            UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=256,
                out_channels=256,
                dropout=0,
            ),
            Rearrange(
                "b (c size_z size_y size_x) n_patch_z n_patch_y n_patch_x -> b c (n_patch_z size_z) (n_patch_y size_y) (n_patch_x size_x)",
                n_patch_z=num_patches[0],
                n_patch_y=num_patches[1],
                n_patch_x=num_patches[2],
                size_y=patch_size[1],
                size_x=patch_size[2],
                size_z=patch_size[0],
            ),
        )

    def forward(self, img):
        features = self.encoder(img)
        features = self.decoder(features)
        return features


class IWMPredictor(torch.nn.Module):
    def __init__(
        self,
        domains: List[str],
        num_patches: List[int],
        input_dim: Optional[int] = 192,
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 12,
        num_head: Optional[int] = 3,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int]
            Number of patches in each dimension
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

        self.domain_embeddings = torch.nn.ParameterDict(
            {d: torch.nn.Parameter(torch.zeros(1, 1, emb_dim)) for d in domains}
        )
        self.context_mixer = torch.nn.Linear(2 * emb_dim, emb_dim, 1)

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(np.prod(num_patches), 1, emb_dim))

        self.predictor_embed = torch.nn.Linear(input_dim, emb_dim)

        self.projector_embed = torch.nn.Linear(emb_dim, input_dim)
        self.norm = torch.nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, context_emb, target_masks, target_domain):
        t, b = target_masks.shape

        # map context embedding to predictor dimension
        context_emb = self.predictor_embed(context_emb)

        # add target domain information via concatenation + token mixing
        target_domain_embedding = torch.cat(
            [self.domain_embeddings[td] for td in target_domain]
        ).repeat(b, context_emb.shape[1], 1)
        context_emb = torch.cat([context_emb, target_domain_embedding], dim=-1)
        context_emb = self.context_mixer(context_emb)

        # add masked positional embedding to mask tokens
        mask = self.mask_token.expand(t, b, -1)
        pe = self.pos_embedding.expand(-1, b, -1)
        pe = take_indexes(pe, target_masks)
        mask = mask + pe
        mask = rearrange(mask, "t b c -> b t c")
        # cross attention from mask tokens to context embedding
        for transformer in self.transformer:
            mask = transformer(mask, context_emb)
        # norm and project back to input dimension
        mask = self.projector_embed(self.norm(mask))
        return mask
