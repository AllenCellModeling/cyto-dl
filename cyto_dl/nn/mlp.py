from typing import Optional

import torch
from torch import nn


def _make_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(),
    )


class MLP(nn.Module):
    def __init__(
        self,
        *dims,
        hidden_layers=[256],
        scale_output: Optional[int] = 1,
    ):
        super().__init__()

        assert len(dims) >= 2

        self.input_dims = dims[:-1]
        self.output_dim = dims[-1] * scale_output

        self.hidden_layers = hidden_layers

        net = [_make_block(sum(self.input_dims), hidden_layers[0])]

        net += [  # noqa: FURB140
            _make_block(input_dim, output_dim)
            for (input_dim, output_dim) in zip(hidden_layers[0:], hidden_layers[1:])
        ]

        net += [nn.Linear(hidden_layers[-1], self.output_dim)]

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
