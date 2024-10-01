# modified from https://github.com/IcarusWizard/MAE/blob/main/model.py#L124

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from cyto_dl.nn.vits.decoder import CrossMAE_Decoder, MAE_Decoder
from cyto_dl.nn.vits.encoder import HieraEncoder, MAE_Encoder
from cyto_dl.nn.vits.utils import match_tuple_dimensions


class MAE_Base(torch.nn.Module, ABC):
    def __init__(
        self, spatial_dims, num_patches, patch_size, mask_ratio, features_only, context_pixels
    ):
        super().__init__()
        num_patches, patch_size, context_pixels = match_tuple_dimensions(
            spatial_dims, [num_patches, patch_size, context_pixels]
        )

        self.spatial_dims = spatial_dims
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.features_only = features_only
        self.context_pixels = context_pixels

    # encoder and decoder must be defined in subclasses
    @property
    @abstractmethod
    def encoder(self):
        pass

    @property
    @abstractmethod
    def decoder(self):
        pass

    def init_encoder(self):
        raise NotImplementedError

    def init_decoder(self):
        raise NotImplementedError

    def forward(self, img):
        features, mask, forward_indexes, backward_indexes = self.encoder(img, self.mask_ratio)
        if self.features_only:
            return features
        predicted_img = self.decoder(features, forward_indexes, backward_indexes)
        return predicted_img, mask


class MAE(MAE_Base):
    def __init__(
        self,
        spatial_dims: int = 3,
        num_patches: Optional[List[int]] = 16,
        patch_size: Optional[List[int]] = 4,
        emb_dim: Optional[int] = 768,
        encoder_layer: Optional[int] = 12,
        encoder_head: Optional[int] = 8,
        decoder_layer: Optional[int] = 4,
        decoder_head: Optional[int] = 8,
        decoder_dim: Optional[int] = 192,
        mask_ratio: Optional[int] = 0.75,
        use_crossmae: Optional[bool] = False,
        context_pixels: Optional[List[int]] = 0,
        input_channels: Optional[int] = 1,
        features_only: Optional[bool] = False,
        learnable_pos_embedding: Optional[bool] = True,
    ) -> None:
        """
        Parameters
        ----------
        spatial_dims: int
            Number of spatial dimensions
        num_patches: List[int]
            Number of patches in each dimension (ZYX order)
        patch_size: List[int]
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
            Number of input channels
        features_only: bool
            Only use encoder to extract features
        learnable_pos_embedding: bool
            If True, learnable positional embeddings are used. If False, fixed sin/cos positional embeddings. Empirically, fixed positional embeddings work better for brightfield images.
        """
        super().__init__(
            spatial_dims=spatial_dims,
            num_patches=num_patches,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            features_only=features_only,
            context_pixels=context_pixels,
        )

        self._encoder = MAE_Encoder(
            self.num_patches,
            spatial_dims,
            self.patch_size,
            emb_dim,
            encoder_layer,
            encoder_head,
            self.context_pixels,
            input_channels,
            n_intermediate_weights=-1 if not use_crossmae else decoder_layer,
        )

        decoder_class = MAE_Decoder
        if use_crossmae:
            decoder_class = CrossMAE_Decoder
        self._decoder = decoder_class(
            num_patches=self.num_patches,
            spatial_dims=spatial_dims,
            patch_size=self.patch_size,
            enc_dim=emb_dim,
            emb_dim=decoder_dim,
            num_layer=decoder_layer,
            num_head=decoder_head,
            learnable_pos_embedding=learnable_pos_embedding,
        )

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder


class HieraMAE(MAE_Base):
    def __init__(
        self,
        architecture: List[Dict],
        spatial_dims: int = 3,
        num_patches: Optional[Union[int, List[int]]] = 16,
        num_mask_units: Optional[Union[int, List[int]]] = 8,
        patch_size: Optional[Union[int, List[int]]] = 4,
        emb_dim: Optional[int] = 64,
        decoder_layer: Optional[int] = 4,
        decoder_head: Optional[int] = 8,
        decoder_dim: Optional[int] = 192,
        mask_ratio: Optional[int] = 0.75,
        use_crossmae: Optional[bool] = False,
        context_pixels: Optional[List[int]] = 0,
        input_channels: Optional[int] = 1,
        features_only: Optional[bool] = False,
    ) -> None:
        """
        Parameters
        ----------
        architecture: List[Dict]
            List of dictionaries specifying the architecture of the transformer. Each dictionary should have the following keys:
            - repeat: int
                Number of times to repeat the block
            - num_heads: int
                Number of heads in the multihead attention
            - q_stride: List[int]
                Stride for the query in each spatial dimension
            - self_attention: bool
                Whether to use self attention or mask unit attention
        spatial_dims: int
            Number of spatial dimensions
        num_patches: int, List[int]
            Number of patches in each dimension (Z)YX order. If int, the same number of patches is used in each dimension.
        num_mask_units: int, List[int]
            Number of mask units in each dimension (Z)YX order. If int, the same number of mask units is used in each dimension.
        patch_size: int, List[int]
            Size of each patch (Z)YX order. If int, the same patch size is used in each dimension.
        emb_dim: int
            Dimension of embedding
        decoder_layer: int
            Number of layers in the decoder
        decoder_head: int
            Number of heads in the decoder
        decoder_dim: int
            Dimension of the decoder
        mask_ratio: float
            Fraction of mask units to remove
        use_crossmae: bool
            Use CrossMAE-style decoder instead of MAE decoder
        context_pixels: List[int]
            Number of extra pixels around each patch to include in convolutional embedding to encoder dimension.
        input_channels: int
            Number of input channels
        features_only: bool
            Only use encoder to extract features
        """
        super().__init__(
            spatial_dims=spatial_dims,
            num_patches=num_patches,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            features_only=features_only,
            context_pixels=context_pixels,
        )
        num_mask_units = match_tuple_dimensions(self.spatial_dims, [num_mask_units])[0]

        self._encoder = HieraEncoder(
            num_patches=self.num_patches,
            num_mask_units=num_mask_units,
            architecture=architecture,
            emb_dim=emb_dim,
            spatial_dims=self.spatial_dims,
            patch_size=self.patch_size,
            context_pixels=self.context_pixels,
            input_channels=input_channels,
        )
        # "patches" to the decoder are actually mask units, so num_patches is num_mask_units, patch_size is mask unit size
        mask_unit_size = (np.array(self.num_patches) * np.array(self.patch_size)) / np.array(
            num_mask_units
        )

        decoder_class = MAE_Decoder
        if use_crossmae:
            decoder_class = CrossMAE_Decoder

        self._decoder = decoder_class(
            num_patches=num_mask_units,
            spatial_dims=spatial_dims,
            patch_size=mask_unit_size.astype(int),
            enc_dim=self.encoder.final_dim,
            emb_dim=decoder_dim,
            num_layer=decoder_layer,
            num_head=decoder_head,
            has_cls_token=False,
        )

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder
