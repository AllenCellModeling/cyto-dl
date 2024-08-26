from typing import List, Optional, Union

import torch
import torch.nn.functional
from einops import rearrange
from timm.models.layers import trunc_normal_

from cyto_dl.nn.vits.blocks import CrossAttentionBlock
from cyto_dl.nn.vits.utils import (
    get_positional_embedding,
    match_tuple_dimensions,
    take_indexes,
)


class JEPAPredictor(torch.nn.Module):
    """Class for predicting target features from context embedding."""

    def __init__(
        self,
        num_patches: Union[int, List[int]],
        spatial_dims: int = 3,
        input_dim: Optional[int] = 192,
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 12,
        num_head: Optional[int] = 3,
        learnable_pos_embedding: Optional[bool] = True,
    ) -> None:
        """
        Parameters
        ----------
        num_patches: List[int], int
            Number of patches in each dimension. If int, the same number of patches is used for all spatial dimensions
        spatial_dims: int
            Number of spatial dimensions
        input_dim: int
            Dimension of input
        emb_dim: int
            Dimension of embedding
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        learnable_pos_embedding: bool
            If True, learnable positional embeddings are used. If False, fixed sin/cos positional embeddings. Empirically, fixed positional embeddings work better for brightfield images.
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

        num_patches = match_tuple_dimensions(spatial_dims, [num_patches])[0]

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = get_positional_embedding(
            num_patches, emb_dim, use_cls_token=False, learnable=learnable_pos_embedding
        )

        self.predictor_embed = torch.nn.Linear(input_dim, emb_dim)

        self.projector_embed = torch.nn.Linear(emb_dim, input_dim)
        self.norm = torch.nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def predict_target_features(self, context_emb, target_masks):
        t, b = target_masks.shape
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

    def forward(self, context_emb, target_masks):
        # map context embedding to predictor dimension
        context_emb = self.predictor_embed(context_emb)
        target_features = self.predict_target_features(context_emb, target_masks)
        return target_features


class IWMPredictor(JEPAPredictor):
    """Specialized JEPA predictor that can conditionally predict between different domains (e.g.
    predict from brightfield to multiple fluorescent tags)"""

    def __init__(
        self,
        domains: List[str],
        num_patches: Union[int, List[int]],
        spatial_dims: int = 3,
        input_dim: Optional[int] = 192,
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 12,
        num_head: Optional[int] = 3,
    ) -> None:
        """
        Parameters
        ----------
        domains: List[str]
            List of names of target domains
        num_patches: List[int]
            Number of patches in each dimension. If int, the same number of patches is used for all spatial dimensions
        spatial_dims: int
            Number of spatial dimensions
        spatial_dims: int
            Number of spatial dimensions
        emb_dim: int
            Dimension of embedding
        num_layer: int
            Number of transformer layers
        num_head: int
            Number of heads in transformer
        """
        super().__init__(
            num_patches=num_patches,
            spatial_dims=spatial_dims,
            input_dim=input_dim,
            emb_dim=emb_dim,
            num_layer=num_layer,
            num_head=num_head,
        )

        self.domain_embeddings = torch.nn.ParameterDict(
            {d: torch.nn.Parameter(torch.zeros(1, 1, emb_dim)) for d in domains}
        )
        self.context_mixer = torch.nn.Linear(2 * emb_dim, emb_dim, 1)

    def forward(self, context_emb, target_masks, target_domain):
        _, b = target_masks.shape
        if len(target_domain) == 1:
            target_domain = target_domain * b
        # map context embedding to predictor dimension
        context_emb = self.predictor_embed(context_emb)

        # add target domain information via concatenation + token mixing
        target_domain_embedding = torch.cat(
            [self.domain_embeddings[td] for td in target_domain]
        ).repeat(1, context_emb.shape[1], 1)
        context_emb = torch.cat([context_emb, target_domain_embedding], dim=-1)
        context_emb = self.context_mixer(context_emb)

        target_features = self.predict_target_features(context_emb, target_masks)
        return target_features
