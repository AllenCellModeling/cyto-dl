import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FoldingNet(nn.Module):
    def __init__(
        self, input_dim: int, num_output_points: int, hidden_dim: int = 512, std: float = 0.3
    ):

        super().__init__()

        if np.sqrt(num_output_points) ** 2 != num_output_points:
            raise ValueError("The number of output points must have an integer square root.")

        self.input_dim = input_dim
        self.num_output_points = num_output_points

        # make grid
        grid_side = np.sqrt(num_output_points).astype(int)
        range_x = torch.linspace(-std, std, grid_side)
        range_y = torch.linspace(-std, std, grid_side)
        xy = torch.meshgrid(range_x, range_y, indexing="ij")
        self.grid = nn.Parameter(xy.float().reshape(-1, 2), requires_grad=False)

        self.hidden_dim = hidden_dim

        if input_dim != hidden_dim:
            self.project = nn.Linear(input_dim, hidden_dim)
        else:
            self.project = nn.Identity()

        self.folding1 = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.folding2 = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x, grid):
        x = x.project(x)

        grid = grid.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = x.unsqueeze(1)
        cw_exp = x.expand(-1, grid.shape[1], -1)

        cat1 = torch.cat((cw_exp, grid), dim=2)
        folding_result1 = self.folding1(cat1)
        cat2 = torch.cat((cw_exp, folding_result1), dim=2)
        folding_result2 = self.folding2(cat2)
        return folding_result2
