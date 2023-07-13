"""
Adapted from: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/decoders.py
License: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/LICENSE_TearingNet
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FoldingNet(nn.Module):
    def __init__(
        self, input_dim: int, 
        num_output_points: int, 
        hidden_dim: int = 512, 
        std: float = 0.3,
        shape: str = 'plane',
        sphere_path: str = "",
        gaussian_path: str = "",
        num_coords: int = 3,
    ):

        super().__init__()

        # if np.sqrt(num_output_points) ** 2 != num_output_points:
        #     raise ValueError("The number of output points must have an integer square root.")

        self.input_dim = input_dim
        self.num_output_points = num_output_points
        self.shape = shape
        self.sphere_path = sphere_path
        self.num_coords = num_coords
        self.gaussian_path = gaussian_path

        # make grid
        if self.shape == 'plane':
            self.grid_dim = 2
            grid_side = np.sqrt(num_output_points).astype(int)
            range_x = torch.linspace(-std, std, grid_side)
            range_y = torch.linspace(-std, std, grid_side)
            # xy = torch.meshgrid(range_x, range_y, indexing="ij")
            # xy = torch.cat(xy, axis=0)
            # self.grid = nn.Parameter(xy.float().reshape(-1, 2), requires_grad=False)
            x_coor, y_coor = torch.meshgrid(range_x, range_y, indexing="ij")
            self.grid = (
                torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)
            )
        elif self.shape == 'sphere':
            self.grid_dim = 3
            self.grid = torch.tensor(np.load(self.sphere_path)).float()
        elif self.shape == 'gaussian':
            self.grid_dim = 3
            self.grid = torch.tensor(np.load(self.gaussian_path)).float()


        self.hidden_dim = hidden_dim

        if input_dim != hidden_dim:
            self.project = nn.Linear(input_dim, hidden_dim, bias=False)
        else:
            self.project = nn.Identity()

        # self.folding1 = nn.Sequential(
        #     nn.Conv1d(hidden_dim + 2, hidden_dim, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_dim, 3, kernel_size=1),
        # )

        # self.folding2 = nn.Sequential(
        #     nn.Conv1d(hidden_dim + 3, hidden_dim, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_dim, 3, kernel_size=1),
        # )
        self.folding1 = nn.Sequential(
            nn.Linear(hidden_dim + self.grid_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_coords),
        )

        self.folding2 = nn.Sequential(
            nn.Linear(hidden_dim + self.num_coords, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_coords),
        )

    def forward(self, x):
        x = self.project(x)

        grid = self.grid.unsqueeze(0).expand(x.shape[0], -1, -1)
        grid = grid.type_as(x)
        x = x.unsqueeze(1)
        cw_exp = x.expand(-1, grid.shape[1], -1)

        cat1 = torch.cat((cw_exp, grid), dim=2)
        folding_result1 = self.folding1(cat1)
        cat2 = torch.cat((cw_exp, folding_result1), dim=2)
        folding_result2 = self.folding2(cat2)

        # cat1 = torch.cat((cw_exp, grid), dim=2)

        # folding_result1 = self.folding1(cat1.transpose(1, -1))
        # cat2 = torch.cat((cw_exp.transpose(1, -1), folding_result1), dim=1)
        # folding_result2 = self.folding2(cat2)
        # return folding_result2.transpose(1, -1)
        return folding_result2
