import torch.nn as nn


class TorchVisionWrapper(nn.Module):
    VALID_MODELS = (
        "ShuffleNetV2",
        "ConvNeXt",
        "QuantizableShuffleNetV2",
        "MaxVit",
        "MobileNetV2",
        "QuantizableResNet",
        "QuantizableMobileNetV2",
        "VGG",
        "GoogLeNet",
        "FCN",
        "MNASNet",
        "SwinTransformer",
        "SqueezeNet",
        "VisionTransformer",
        "AlexNet",
        "DenseNet",
        "QuantizableGoogLeNet",
        "ResNet",
        "LRASPP",
        "QuantizableMobileNetV3",
        "Inception3",
        "QuantizableInception3",
        "RegNet",
        "MobileNetV3",
        "EfficientNet",
    )

    def __init__(self, base_encoder, in_channels=1):
        """Wrap a torchvision model to accept a different number of input channels.

        Parameters
        ----------
        base_encoder:
            An initialized torchvision model. The following models are supported: ShuffleNetV2, ConvNeXt, QuantizableShuffleNetV2, MaxVit, MobileNetV2, QuantizableResNet, QuantizableMobileNetV2, VGG, GoogLeNet, FCN, MNASNet, SwinTransformer, SqueezeNet, VisionTransformer, AlexNet, DenseNet, QuantizableGoogLeNet, ResNet, LRASPP, QuantizableMobileNetV3, Inception3, QuantizableInception3, RegNet, MobileNetV3, EfficientNet
        in_channels:
            number of input channels (default: 1)
        """
        if base_encoder.__class__.__name__ not in self.VALID_MODELS:
            raise ValueError(
                f"Model {base_encoder.__class__.__name__} not supported, only {self.VALID_MODELS} are supported"
            )
        super().__init__()
        if in_channels != 3:
            # find first Conv2D with 3 input channels
            for layer in base_encoder.modules():
                if isinstance(layer, nn.Conv2d) and layer.in_channels == 3:
                    new_layer = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                        bias=layer.bias is not None,
                        padding_mode=layer.padding_mode,
                    )
                    # Replace the old layer with the new layer
                    layer.weight = new_layer.weight
                    layer.bias = new_layer.bias
                    layer.in_channels = in_channels
                    break
            else:
                raise ValueError(
                    "Could not find Conv2D layer with 3 input channels. Please create a GitHub issue or provide one of the valid models."
                )
        self.encoder = base_encoder

    def forward(self, x):
        return self.encoder(x)
