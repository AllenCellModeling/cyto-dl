import torch.nn as nn

_full_slice = slice(None, None, None)


class Prior(nn.Module):
    def __init__(self, dimensionality: int, **kwargs):
        super().__init__()
        self.dimensionality = dimensionality

    def __len__(self):
        return self.dimensionality

    @property
    def param_size(self):
        raise NotImplementedError

    def kl_divergence(self, z):
        raise NotImplementedError

    def sample(self, z):
        raise NotImplementedError

    def forward(self, z, mode="kl", inference=False, **kwargs):
        raise NotImplementedError
