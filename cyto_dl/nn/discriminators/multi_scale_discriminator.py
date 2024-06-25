from torch import nn

from .n_layer_discriminator import NLayerDiscriminator


class MultiScaleDiscriminator(nn.Module):
    """Modified version of Pix2PixHD discriminator, which returns discriminator activations at
    multiple spatial scales."""

    def __init__(self, n_scales: int = 2, dim: int = 3, **kwargs):
        """
        Parameters
        ----------
            n_scales:int=2
                Number of spatial scales to
            **kwargs
                Arguments to pass to NLayerDiscriminator
        """
        super().__init__()
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        self.scales = range(n_scales)
        kwargs.update({"dim": dim})
        self.discriminators = nn.ModuleDict(
            {str(scale): NLayerDiscriminator(**kwargs) for scale in self.scales}
        )
        self.pooling_fn = nn.AvgPool3d if dim == 3 else nn.AvgPool2d

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
                source_img = self.pooling_fn(
                    kernel_size=3, padding=1, stride=2, count_include_pad=False
                )(source_img)
                img = self.pooling_fn(kernel_size=3, padding=1, stride=2, count_include_pad=False)(
                    img
                )
            features[key] = result
        return features
