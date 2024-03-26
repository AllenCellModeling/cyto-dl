from typing import List, Optional, Union

import numpy as np
import torch
from einops.layers.torch import Rearrange
from monai.networks.blocks import UnetOutBlock, UnetResBlock, UpSample

from cyto_dl.nn.vits.mae import MAE_Encoder


class EncodedSkip(torch.nn.Module):
    def __init__(self, spatial_dims, num_patches, emb_dim, n_decoder_filters, layer):
        super().__init__()
        """
        layer = 0 is the smallest resolution, n is the highest
        as the layer increases, the image size increases and the number of filters decreases
        """
        upsample = 2**layer
        self.n_out_channels = n_decoder_filters // (upsample**spatial_dims)
        self.patch2img = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, n_decoder_filters),
            torch.nn.LayerNorm(n_decoder_filters),
            Rearrange(
                " (n_patch_z n_patch_y n_patch_x) b (c uz uy ux) ->  b c (n_patch_z uz) (n_patch_y uy) (n_patch_x ux)",
                n_patch_z=num_patches[0],
                n_patch_y=num_patches[1],
                n_patch_x=num_patches[2],
                c=self.n_out_channels,
                uz=upsample,
                uy=upsample,
                ux=upsample,
            ),
        )

    def forward(self, features):
        return self.patch2img(features)


class SuperresDecoder_efficient_multilayer(torch.nn.Module):
    """create unet-like decoder where each decoder layer is a fed a skip connection consisting of a
    different weighted sum of intermediate layer features."""

    def __init__(
        self,
        spatial_dims: int = 3,
        num_patches: Optional[List[int]] = [2, 32, 32],
        base_patch_size: Optional[List[int]] = [4, 8, 8],
        emb_dim: Optional[int] = 192,
        n_decoder_filters: Optional[int] = 16,
        out_channels: Optional[int] = 6,
        upsample_factor: Optional[Union[int, List[int]]] = [2.6134, 2.5005, 2.5005],
        num_layer: Optional[int] = 3,
    ) -> None:
        super().__init__()

        total_upsample_factor = np.array(upsample_factor) * np.array(base_patch_size)
        self.num_layer = np.min(np.log2(total_upsample_factor)).astype(int)
        residual_resize_factor = list(total_upsample_factor / 2**num_layer)

        input_n_decoder_filters = n_decoder_filters

        self.upsampling = torch.nn.ModuleDict()
        for i in range(self.num_layer):
            skip = EncodedSkip(spatial_dims, num_patches, emb_dim, input_n_decoder_filters, i)
            n_input_channels = (
                n_decoder_filters + skip.n_out_channels if i > 0 else n_decoder_filters
            )
            self.upsampling[f"layer_{i}"] = torch.nn.ModuleDict(
                {
                    "skip": skip,
                    "upsample": torch.nn.Sequential(
                        *[
                            UnetResBlock(
                                spatial_dims=3,
                                in_channels=n_input_channels,
                                out_channels=n_decoder_filters // 2,
                                stride=1,
                                kernel_size=3,
                                norm_name="INSTANCE",
                                dropout=0,
                            ),
                            # no convolution in upsample, do convolution at low resolution
                            UpSample(
                                spatial_dims=3,
                                in_channels=n_decoder_filters // 2,
                                out_channels=n_decoder_filters // 2,
                                scale_factor=[2, 2, 2],
                                mode="nontrainable",
                            ),
                        ]
                    ),
                }
            )
            n_decoder_filters = n_decoder_filters // 2

        skip = EncodedSkip(
            spatial_dims, num_patches, emb_dim, input_n_decoder_filters, self.num_layer
        )
        n_input_channels = n_decoder_filters + skip.n_out_channels
        self.upsampling[f"layer_{i+1}"] = torch.nn.ModuleDict(
            {
                "skip": skip,
                "upsample": torch.nn.Sequential(
                    *[
                        UpSample(
                            spatial_dims=spatial_dims,
                            in_channels=n_input_channels,
                            out_channels=n_decoder_filters // 2,
                            scale_factor=residual_resize_factor,
                            mode="nontrainable",
                            interp_mode="trilinear",
                        ),
                        UnetResBlock(
                            spatial_dims=spatial_dims,
                            in_channels=n_decoder_filters // 2,
                            out_channels=n_decoder_filters // 2,
                            stride=1,
                            kernel_size=3,
                            norm_name="INSTANCE",
                            dropout=0,
                        ),
                        UnetOutBlock(
                            spatial_dims=spatial_dims,
                            in_channels=n_decoder_filters // 2,
                            out_channels=out_channels,
                            dropout=0,
                        ),
                    ]
                ),
            }
        )

    def forward(
        self,
        features: torch.Tensor,
    ):
        # remove global token
        features = features[:, 1:]
        prev = None
        for i in range(self.num_layer + 1):
            skip = self.upsampling[f"layer_{i}"]["skip"](features[i])
            if prev is not None:
                skip = torch.cat([skip, prev], dim=1)
            prev = self.upsampling[f"layer_{i}"]["upsample"](skip)
        return prev


class SuperresDecoder_efficient(torch.nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        num_patches: Optional[List[int]] = [2, 32, 32],
        base_patch_size: Optional[List[int]] = [4, 8, 8],
        emb_dim: Optional[int] = 192,
        n_decoder_filters: Optional[int] = 16,
        out_channels: Optional[int] = 6,
        upsample_factor: Optional[Union[int, List[int]]] = [2.6134, 2.5005, 2.5005],
        num_layer: Optional[int] = 3,
    ) -> None:
        super().__init__()
        self.patch2img = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, n_decoder_filters),
            torch.nn.LayerNorm(n_decoder_filters),
            Rearrange(
                " (n_patch_z n_patch_y n_patch_x) b c ->  b c n_patch_z n_patch_y n_patch_x",
                n_patch_z=num_patches[0],
                n_patch_y=num_patches[1],
                n_patch_x=num_patches[2],
                c=n_decoder_filters,
            ),
        )
        # patch 2 img forms an img where each token is one pixl, we want to upsample this to the original image size * upsample factor
        total_upsample_factor = np.array(upsample_factor) * np.array(base_patch_size)
        num_layer = np.min(np.log2(total_upsample_factor)).astype(int)
        residual_resize_factor = list(total_upsample_factor / 2**num_layer)

        upsampling = []
        for i in range(num_layer):
            n_decoder_filters = n_decoder_filters // 2 if i > 0 else n_decoder_filters
            upsampling += [
                UnetResBlock(
                    spatial_dims=3,
                    in_channels=n_decoder_filters,
                    out_channels=n_decoder_filters // 2,
                    stride=1,
                    kernel_size=3,
                    norm_name="INSTANCE",
                    dropout=0,
                ),
                UpSample(
                    spatial_dims=3,
                    in_channels=n_decoder_filters // 2,
                    out_channels=n_decoder_filters // 2,
                    scale_factor=[2, 2, 2],
                    mode="pixelshuffle",
                ),
            ]
        upsampling += [
            UpSample(
                spatial_dims=spatial_dims,
                in_channels=n_decoder_filters // 2,
                out_channels=n_decoder_filters // 2,
                scale_factor=residual_resize_factor,
                mode="nontrainable",
                interp_mode="trilinear",
            ),
            UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=n_decoder_filters // 2,
                out_channels=n_decoder_filters // 2,
                stride=1,
                kernel_size=3,
                norm_name="INSTANCE",
                dropout=0,
            ),
            UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=n_decoder_filters // 2,
                out_channels=out_channels,
                dropout=0,
            ),
        ]
        self.upsampling = torch.nn.Sequential(*upsampling)

    def forward(
        self,
        features: torch.Tensor,
    ):
        # remove global token
        features = features[1:]
        img = self.patch2img(features)
        img = self.upsampling(img)
        return img


class SuperresDecoder(torch.nn.Module):
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

        self.lr_conv = torch.nn.Sequential(
            *[
                UnetResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=n_decoder_filters,
                    out_channels=n_decoder_filters,
                    stride=1,
                    kernel_size=3,
                    norm_name="INSTANCE",
                    dropout=0,
                )
                for _ in range(num_layer)
            ]
        )

        self.upsampler = torch.nn.Sequential(
            UpSample(
                spatial_dims=spatial_dims,
                in_channels=n_decoder_filters,
                out_channels=n_decoder_filters,
                scale_factor=np.array(upsample_factor),
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

        out_dim = torch.prod(torch.as_tensor(base_patch_size)) * n_decoder_filters
        self.head = torch.nn.Linear(emb_dim, out_dim)
        self.head_norm = torch.nn.LayerNorm(out_dim.item())

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
                c=n_decoder_filters,
            )
        elif spatial_dims == 2:
            self.patch2img = Rearrange(
                "(n_patch_y n_patch_x) b (c patch_size_y patch_size_x) ->  b c (n_patch_y patch_size_y) (n_patch_x patch_size_x)",
                n_patch_y=num_patches[0],
                n_patch_x=num_patches[1],
                patch_size_y=base_patch_size[0],
                patch_size_x=base_patch_size[1],
                c=n_decoder_filters,
            )

    def forward(self, features):
        # remove global feature
        features = features[1:]

        # (npatches x npatches x npatches) b (emb dim) -> (npatches* npatches * npatches) b (c z y x)
        patches = self.head_norm(self.head(features))

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
        decoder_layer: Optional[int] = 3,
        n_decoder_filters: Optional[int] = 16,
        out_channels: Optional[int] = 6,
        upsample_factor: Optional[List[int]] = [2.6134, 2.5005, 2.5005],
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
            **encoder_kwargs,
        )
        if encoder_ckpt is not None:
            model = torch.load(encoder_ckpt)
            enc_state_dict = {
                k.replace("backbone.encoder.", ""): v
                for k, v in model["state_dict"].items()
                if "encoder" in k
            }
            self.encoder.load_state_dict(enc_state_dict, strict=False)

        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                # allow different weighting of internal activations for finetuning
                param.requires_grad = "intermediate_weighter" in name

        self.decoder = SuperresDecoder_efficient_multilayer(
            spatial_dims=spatial_dims,
            num_patches=num_patches,
            base_patch_size=base_patch_size,
            emb_dim=emb_dim,
            num_layer=decoder_layer,
            n_decoder_filters=n_decoder_filters,
            out_channels=out_channels,
            upsample_factor=upsample_factor,
        )

    def forward(self, img):
        features = self.encoder(img, mask_ratio=0)
        return self.decoder(features)
