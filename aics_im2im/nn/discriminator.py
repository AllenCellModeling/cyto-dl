import functools

import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator."""

    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.InstanceNorm3d,
        noise_annealer=None,
    ):
        """Construct a PatchGAN discriminator.

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        modules = nn.ModuleDict()

        kw = 4
        padw = 1
        modules.update(
            {
                "block_0": nn.Sequential(
                    nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
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
                        nn.Conv3d(
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
                f"block_{n+1}": nn.Sequential(
                    nn.Conv3d(
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
                f"block_{n+2}": nn.Sequential(
                    nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
                )
            }
        )
        self.model = modules
        self.noise_annealer = noise_annealer

    def set_requires_grad(self, requires_grad=False):
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def forward(self, im, x, requires_features=False):
        """Standard forward."""
        output_im = [x]
        if self.noise_annealer is not None:
            im = self.noise_annealer(im)
        output_im = torch.cat([x, im], 1)
        if requires_features:
            results = [output_im]
            for k, v in self.model.items():
                results.append(v(results[-1]))
            return results[1:]
        return nn.Sequential(*self.model)(output_im)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_scales, **kwargs):
        super().__init__()
        self.scales = range(n_scales)
        self.discriminators = nn.ModuleDict(
            {str(scale): NLayerDiscriminator(**kwargs) for scale in self.scales}
        )

    def set_requires_grad(self, requires_grad=False):
        for scale in self.scales:
            self.discriminators[str(scale)].set_requires_grad(requires_grad)

    def forward(self, x, real, pred):
        features = {}
        for key, img in zip(["real", "pred"], [real, pred]):
            result = {}
            source_img = x.detach().clone()
            for scale in self.scales:
                result[scale] = self.discriminators[str(scale)](
                    img,
                    source_img,
                    requires_features=True,
                )
                source_img = nn.AvgPool3d(
                    kernel_size=3, padding=1, stride=2, count_include_pad=False
                )(source_img)
                img = nn.AvgPool3d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(
                    img
                )

            features[key] = result
        return features
