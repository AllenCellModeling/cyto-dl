import functools

import torch
from torch import nn


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator."""

    def __init__(
        self,
        input_nc: int = 2,
        ndf: int = 64,
        n_layers: int = 3,
        dim: int = 3,
        norm_layer=nn.InstanceNorm3d,
        noise_annealer=None,
    ):
        """
        Parameters
        ----------
            input_nc:int=2
                Number of channels of input images. Generally, `n_channels(input_im)+n_channels(model_output)`
            ndf:int=64
                Number of filters in the first conv_fn layer. Later layers are multiples of `ndf`.
            n_layers:int=3
                Number of conv_fn layers in the discriminator
            norm_layer=nn.InstanceNorm3d
                normalization layer
            noise_annealer=None
                Noise annealer object
        """
        super().__init__()
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        conv_fn = nn.Conv3d if dim == 3 else nn.Conv2d

        modules = nn.ModuleDict()

        kw = 4
        padw = 1
        modules.update(
            {
                "block_0": nn.Sequential(
                    conv_fn(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True),
                )
            }
        )
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            modules.update(
                {
                    f"block_{n}": nn.Sequential(
                        conv_fn(
                            ndf * nf_mult_prev,
                            ndf * nf_mult,
                            kernel_size=kw,
                            stride=2,
                            padding=padw,
                            bias=use_bias,
                        ),
                        norm_layer(ndf * nf_mult),
                        nn.LeakyReLU(0.2, True),
                    )
                }
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        modules.update(
            {
                f"block_{n_layers}": nn.Sequential(
                    conv_fn(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=1,
                        padding=padw,
                        bias=use_bias,
                    ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                )
            }
        )

        modules.update(
            {
                f"block_{n_layers+1}": nn.Sequential(
                    conv_fn(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
                )
            }
        )
        self.model = modules
        self.noise_annealer = noise_annealer

    def forward(self, im, x, requires_features=False):
        """Standard forward."""
        if self.noise_annealer is not None:
            im = self.noise_annealer(im)
        if x.shape != im.shape:
            x = torch.nn.functional.interpolate(
                input=x,
                size=im.shape[2:],
            )
        output_im = torch.cat([x, im], 1)
        if requires_features:
            results = [output_im]
            for k, v in self.model.items():
                results.append(v(results[-1]))
            return results[1:]
        return nn.Sequential(*self.model)(output_im)
