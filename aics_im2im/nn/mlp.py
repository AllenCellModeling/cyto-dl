from typing import Optional

from torch import nn


def _make_block(input_dim, output_dim, layer_norm=True):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim) if layer_norm else nn.Identity(),
        nn.SiLU(),
    )


class MLP(nn.Module):
    def __init__(
        self,
        *dims,
        hidden_layers=[256],
        scale_output: Optional[int] = 1,
        layer_norm: bool = True,
    ):
        super().__init__()

        assert len(dims) >= 2

        self.input_dims = dims[:-1]
        self.output_dim = dims[-1] * scale_output

        self.hidden_layers = hidden_layers

        net = [_make_block(sum(self.input_dims), hidden_layers[0], layer_norm)]

        net += [
            _make_block(input_dim, output_dim)
            for (input_dim, output_dim) in zip(hidden_layers[0:], hidden_layers[1:])
        ]

        net += [nn.Linear(hidden_layers[-1], self.output_dim)]

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
