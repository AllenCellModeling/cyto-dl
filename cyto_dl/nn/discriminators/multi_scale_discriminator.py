from torch import nn

from .n_layer_discriminator import NLayerDiscriminator


class MultiScaleDiscriminator(nn.Module):
    """Modified version of Pix2PixHD discriminator, which returns discriminator activations at
    multiple spatial scales."""

    def __init__(self, n_scales: int = 2, **kwargs):
        """
        Parameters
        ----------
            n_scales:int=2
                Number of spatial scales to
            **kwargs
                Arguments to pass to NLayerDiscriminator
        """
        super().__init__()
        self.scales = range(n_scales)
        self.discriminators = nn.ModuleDict(
            {str(scale): NLayerDiscriminator(**kwargs) for scale in self.scales}
        )

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
                img = nn.AvgPool3d(
                    kernel_size=3, padding=1, stride=2, count_include_pad=False
                )(img)

            features[key] = result
        return features
