import torch

class FlexibleTorchvision(torch.nn.Module):
    def __init__(self, in_channels, torchvision_model):
        super().__init__()
        if hasattr(torchvision_model, 'conv1'):
            torchvision_model.conv1 = torch.nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )   
        else:
            raise NotImplementedError("This model doesn't have a conv1 layer")
        self.model = torchvision_model

    def forward(self, x):
        return self.model(x)