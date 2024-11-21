import torch.nn as nn

class ResNet(nn.Module):
    """
    Resnet wrapper for variable input channels and output dimensions
    """
    def __init__(self, base_encoder, num_channels=1, out_dim=2048, zero_init_residual=True, normalize_embeddings: bool = False, **kwargs):
        """
        Parameters
        ----------
        base_encoder: 
            resnet model from torchviion
        num_channels: 
            number of input channels (default: 1)
        out_dim: 
            feature dimension (default: 2048)
        zero_init_residual: 
            zero-initialize the last BN in each residual branch (default: True)
        normalize_embeddings:
            whether to normalize embeddings (default: False)
        kwargs:
            additional arguments to pass to the base encoder.Full arguments can be found [here](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
        """
        super().__init__()
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        kwargs['num_classes'] = out_dim
        kwargs['zero_init_residual'] = zero_init_residual
        self.encoder = base_encoder(**kwargs)
        # make first layer of encoder match num_channels
        self.encoder.conv1 = nn.Conv2d(num_channels, 
            self.encoder.conv1.out_channels,
            kernel_size=self.encoder.conv1.kernel_size,
            stride=self.encoder.conv1.stride,
            padding=self.encoder.conv1.padding,
            bias=False
        )
        self.normalize_embeddings = normalize_embeddings

    def forward(self, x1):
        emb = self.encoder(x1)
        if self.normalize_embeddings:
            emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb