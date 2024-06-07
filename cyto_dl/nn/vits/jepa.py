from typing import List, Optional
import torch
from einops import rearrange
from timm.models.vision_transformer import Block
from cyto_dl.nn.vits.blocks import Patchify
import numpy as np
from cyto_dl.nn.vits.blocks import CrossAttentionBlock
from cyto_dl.nn.vits.utils import take_indexes
from timm.models.layers import trunc_normal_

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

        