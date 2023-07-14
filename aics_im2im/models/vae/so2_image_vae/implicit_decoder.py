from typing import Optional, Sequence

import torch
import torch.nn as nn


def make_parallelepipeds(input_dims, batch_size=0):

    ranges = [torch.linspace(-1, 1, dim) for dim in input_dims]

    ppiped = torch.stack(torch.meshgrid(*ranges))

    if batch_size:
        repeats = (len(input_dims) + 1) * [1]
        return ppiped.unsqueeze(0).repeat(batch_size, *repeats)
    return ppiped


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        up_conv: bool = False,
        non_linearity: Optional[nn.Module] = None,
        skip_connection: bool = False,
        batch_norm: bool = False,
        mode: str = "3d",
    ):
        """Convolutional block class.
        Parameters
        ----------
        in_c: int
            number of input channels
        out_c: int
            number of output channels
        kernel_size: Sequence[int] (defaults to (3, 3, 3))
            dimensions of the convolutional kernel to be applied
        stride: Sequence[int] (defaults to (1, 1, 1))
            stride of the convolution
        padding:
            padding value for the convolution (defaults to 0)
        mode:
            Dimensionality of the input data. Can be "2d" or "3d".
        """

        super().__init__()
        self.skip_connection = skip_connection
        self.block = conv_block(
            in_c=in_c,
            out_c=out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            up_conv=up_conv,
            non_linearity=non_linearity,
            batch_norm=batch_norm,
            mode=mode,
        )

    def forward(self, x):
        res = self.block(x)
        if self.skip_connection and (res.shape[1] == x.shape[1]):
            return res + nn.functional.interpolate(x, res.shape[2:])
        else:
            return res


def conv_block(
    in_c: int,
    out_c: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    up_conv: bool = False,
    non_linearity: Optional[nn.Module] = None,
    batch_norm: bool = True,
    mode: str = "3d",
):
    """Util function to instantiate a convolutional block.
    Parameters
    ----------
    in_c: int
        number of input channels
    out_c: int
        number of output channels
    kernel_size: Sequence[int] (defaults to (3, 3, 3))
        dimensions of the convolutional kernel to be applied
    stride: Sequence[int] (defaults to (1, 1, 1))
        stride of the convolution
    padding:
        padding value for the convolution (defaults to 0)
    mode:
        Dimensionality of the input data. Can be "2d" or "3d".
    """
    batch_norm_cls = nn.Identity
    if mode == "2d":
        conv = nn.ConvTranspose2d if up_conv else nn.Conv2d
        if batch_norm:
            batch_norm_cls = nn.BatchNorm2d
    elif mode == "3d":
        conv = nn.ConvTranspose3d if up_conv else nn.Conv3d
        if batch_norm:
            batch_norm_cls = nn.BatchNorm3d
    else:
        raise ValueError(f"Mode must be '2d' or '3d'. You passed '{mode}'")

    if non_linearity is None:
        non_linearity = nn.ReLU()

    block = nn.Sequential(
        conv(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
        non_linearity,
        batch_norm_cls(out_c),
    )

    return block


class ImplicitDecoder(nn.Module):
    def __init__(
        self,
        latent_dims,
        hidden_channels: Sequence[int],
        non_linearity: Optional[nn.Module] = None,
        final_non_linearity: Optional[nn.Module] = None,
        mode: str = "3d",
    ):
        super().__init__()
        self.mode = mode
        self._mode = 3 if mode == "3d" else 2
        self.latent_dims = latent_dims

        if non_linearity is None:
            non_linearity = nn.ReLU()

        if final_non_linearity is None:
            final_non_linearity = nn.Identity()

        self.final_non_linearity = final_non_linearity

        layers = []
        _in_channels = latent_dims
        for index, out_channels in enumerate(hidden_channels):
            layers.append(
                conv_block(
                    _in_channels + self._mode + (self.latent_dims if index != 0 else 0),
                    out_channels,
                    kernel_size=1,
                    up_conv=False,
                    non_linearity=non_linearity,
                    mode=mode,
                )
            )
            _in_channels = out_channels

        self.layers = nn.ModuleList(layers)

    def forward(self, z, input_dims=(10, 10, 10), angles=None, translations=None):
        ppipeds = make_parallelepipeds(input_dims, batch_size=z.shape[0])

        if angles is not None:
            pass
            #  rot_matrices = euler_angles_to_matrix(angles, convention="XYZ")
            #  ppipeds = torch.matmul(ppipeds, rot_matrices)

        if translations is not None:
            pass
            #  ppipeds = ppipeds + translations.view(
            #      x.shape[0], 1, *([1] * len(input_dims))
            #  )

        ppipeds = ppipeds.to(z.device)

        z = z.view(*z.shape, *([1] * len(input_dims)))
        z = z.expand(-1, -1, *input_dims)

        y = z
        # Batch,channel (lt dim),input dims
        for index, layer in enumerate(self.layers):
            to_cat = (ppipeds, y) if index == 0 else (ppipeds, z, y)
            res = layer(torch.cat(to_cat, axis=1))
            # if output and input dimensions match
            if res.shape[1] == y.shape[1]:
                # skip connection, excluding the scoordinates and the latent code
                y = res + y
            else:
                y = res

        return self.final_non_linearity(y)
