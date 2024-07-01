import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, base_encoder, num_channels=1, out_dim=2048, zero_init_residual=True):
        """
        base_encoder: resnet model from torchviion
        num_channels: number of input channels (default: 1)
        out_dim: feature dimension (default: 2048)
        """
        super().__init__()
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=out_dim, zero_init_residual=zero_init_residual)
        # make first layer of encoder match num_channels
        self.encoder.conv1 = nn.Conv2d(num_channels, 
            self.encoder.conv1.out_channels,
            kernel_size=self.encoder.conv1.kernel_size,
            stride=self.encoder.conv1.stride,
            padding=self.encoder.conv1.padding,
            bias=False
        )

    def forward(self, x1):
        return self.encoder(x1)
