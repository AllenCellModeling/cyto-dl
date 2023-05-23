from typing import Optional, Sequence

import torch.nn as nn
from monai.networks.layers.simplelayers import Flatten, Reshape
from monai.networks.nets import VarAutoEncoder
from torch.nn.modules.loss import _Loss as Loss

from .koopman_ae import KoopmanAE


class ImageKoopmanAE(KoopmanAE):
    def __init__(
        self,
        latent_dim: int,
        x_label: str,
        in_channels: int,
        channels,
        strides,
        rank: int = 0,
        reconstruction_loss: Loss = nn.MSELoss(reduction="mean"),
        spatial_dims: int = 3,
        kernel_size: int = 3,
        up_kernel_size: int = 3,
        num_res_units: int = 0,
        inter_channels: Optional[Sequence] = None,
        inter_dilations: Optional[Sequence] = None,
        num_inter_units: int = 2,
        act: str = "PRELU",
        norm: str = "INSTANCE",
        dropout: Optional[float] = None,
        bias: bool = True,
        **base_kwargs,
    ):
        """Instantiate an Image Koopman Autoencoder model.

        Parameters
        ----------
        encoder: nn.Module
            Encoder network
        decoder: nn.Module
            Decoder network
        latent_dim: int
            Bottleneck size
        x_label: Optional[str] = None
        reconstruction_loss: Loss
            Loss to be used for reconstruction. Can be a PyTorch loss or a class
            that respects the same interface,
            i.e. subclasses torch.nn.modules._Loss
        in_channels: int
            The number of channels in the input image
        channels: Sequence[int]
            Number of convolutional filters for each layer
        strides: Sequence[int]
            The strides of the convolutional layers
        rank: int = 0
            The rank of the tensor decomposition
        spatial_dims: int = 3
            The number of spatial dimensions in the input
        kernel_size: int = 3
            The size of the convolutional kernels
        up_kernel_size: int = 3
            The size of the upconvolutional kernels
        num_res_units: int = 0
            The number of residual units
        inter_channels: Optional[Sequence] = None
            The number of channels in the intermediate layers
        inter_dilations: Optional[Sequence] = None
            The dilation of the intermediate layers
        num_inter_units: int = 2
            The number of intermediate layers
        act: str = 'PRELU'
            The activation function to use
        norm: str = 'INSTANCE'
            The normalization to use
        dropout: Optional[float] = None
            The amount of dropout to use
        bias: bool = True
            Whether or not to use bias in the convolutional layers
        **base_kwargs:
            Additional arguments passed to BaseModel
        """

        net = VarAutoEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            latent_size=latent_dim,
            out_channels=in_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
            inter_channels=inter_channels,
            inter_dilations=inter_dilations,
            num_inter_units=num_inter_units,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

        del net.logvar

        encoder = nn.Sequential(net.encoder, net.intermediate, Flatten(), net.mu)

        decoder = nn.Sequential(
            net.decodeL,
            nn.ReLU(),
            Reshape(net.channels[-1], *net.final_size),
            net.decoder,
        )

        super().__init__(encoder=encoder, decoder=decoder, **base_kwargs)
