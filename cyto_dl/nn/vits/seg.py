from typing import List, Optional, Union

import torch
from einops.layers.torch import Rearrange
from monai.networks.blocks import UnetOutBlock, UnetResBlock, UpSample

from cyto_dl.nn.vits.mae import MAE_Encoder


class SupperresDecoder(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        num_patches: Optional[List[int]] = [2, 32, 32],
        base_patch_size: Optional[List[int]] = [4, 8, 8],
        emb_dim: Optional[int] = 192,
        num_layer: Optional[int] = 3,
        n_decoder_filters: Optional[int] = 16,
        out_channels: Optional[int] = 6,
        upsample_factor: Optional[Union[int, List[int]]] = [2.6134, 2.5005, 2.5005],
    ) -> None:
        """
        Parameters
        ----------
        spatial_dims: Optional[int]=3
            Number of spatial dimensions
        num_patches: Optional[List[int]]=[2, 32, 32]
            Number of patches in each dimension (ZYX) order
        base_patch_size: Optional[List[int]]=[16, 16, 16]
            Base patch size in each dimension (ZYX) order
        emb_dim: Optional[int] =768
            Embedding dimension of ViT backbone
        num_layer: Optional[int] =3
            Number of layers in convolutional decoder
        n_decoder_filters: Optional[int] =16
            Number of filters in convolutional decoder
        out_channels: Optional[int] =6
            Number of output channels in convolutional decoder. Should be 6 for instance segmentation.
        upsample_factor:Optional[List[int]] = [2.6134, 2.5005, 2.5005]
            Upsampling factor for each dimension (ZYX) order. Default is AICS 20x to 100x objective upsampling
        """
        super().__init__()

        self.lr_conv = []
        for i in range(num_layer):
            if i == 0:
                num_channels = 1
            else:
                num_channels = 16
            self.lr_conv.append(
                UnetResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels,
                    out_channels=n_decoder_filters,
                    stride=1,
                    kernel_size=3,
                    norm_name="INSTANCE",
                    dropout=0,
                ),
            )

        self.lr_conv = torch.nn.Sequential(*self.lr_conv)

        self.upsampler = torch.nn.Sequential(
            UpSample(
                spatial_dims=spatial_dims,
                in_channels=n_decoder_filters,
                out_channels=n_decoder_filters,
                scale_factor=upsample_factor,
                mode="nontrainable",
                interp_mode="trilinear",
            ),
            UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=n_decoder_filters,
                out_channels=n_decoder_filters,
                stride=1,
                kernel_size=3,
                norm_name="INSTANCE",
                dropout=0,
            ),
            UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=n_decoder_filters,
                out_channels=out_channels,
                dropout=0,
            ),
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

    def forward(self, features):
        # remove global feature
        features = features[1:]

        # (npatches x npatches x npatches) b (emb dim) -> (npatches* npatches * npatches) b (z y x)
        patches = self.head(features)

        # patches to image
        img = self.patch2img(patches)

        img = self.lr_conv(img)
        img = self.upsampler(img)
        return img


class Seg_ViT(torch.nn.Module):
    """Class for training a simple convolutional decoder on top of a pretrained ViT backbone."""

    def __init__(
        self,
        spatial_dims: int = 3,
        num_patches: Optional[List[int]] = [2, 32, 32],
        base_patch_size: Optional[List[int]] = [16, 16, 16],
        emb_dim: Optional[int] = 768,
        encoder_layer: Optional[int] = 12,
        encoder_head: Optional[int] = 8,
        decoder_layer: Optional[int] = 3,
        n_decoder_filters: Optional[int] = 16,
        out_channels: Optional[int] = 6,
        mask_ratio: Optional[int] = 0.75,
        upsample_factor: Optional[List[int]] = [2.6134, 2.5005, 2.5005],
        encoder_ckpt: Optional[str] = None,
        freeze_encoder: Optional[bool] = True,
    ) -> None:
        """
        Parameters
        ----------
        spatial_dims: Optional[int]=3
            Number of spatial dimensions
        num_patches: Optional[List[int]]=[2, 32, 32]
            Number of patches in each dimension (ZYX) order
        base_patch_size: Optional[List[int]]=[16, 16, 16]
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
        if isinstance(base_patch_size, int):
            base_patch_size = [base_patch_size] * spatial_dims
        if isinstance(upsample_factor, int):
            upsample_factor = [upsample_factor] * spatial_dims
        assert len(num_patches) == spatial_dims
        assert len(base_patch_size) == spatial_dims
        assert len(upsample_factor) == spatial_dims

        self.encoder = MAE_Encoder(
            spatial_dims=spatial_dims,
            num_patches=num_patches,
            base_patch_size=base_patch_size,
            emb_dim=emb_dim,
            num_layer=encoder_layer,
            num_head=encoder_head,
            mask_ratio=mask_ratio,
        )
        if encoder_ckpt is not None:
            model = torch.load(encoder_ckpt)
            enc_state_dict = {
                k.replace("backbone.encoder.", ""): v
                for k, v in model["state_dict"].items()
                if "encoder" in k
            }

            self.encoder.load_state_dict(enc_state_dict)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.decoder = SupperresDecoder(
            spatial_dims,
            num_patches,
            base_patch_size,
            emb_dim,
            decoder_layer,
            n_decoder_filters,
            out_channels,
            upsample_factor,
        )

    def forward(self, img):
        features = self.encoder(img, do_mask=False)
        return self.decoder(features)
