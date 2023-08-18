import torch
import torch.nn as nn
from scipy.spatial import ConvexHull
import numpy as np


class SDFGridLoss(nn.Module):
    def __init__(
        self,
        n_samples=10,
        max_x=25,
        max_y=25,
        max_z=5,
        grid_size=1000,
        p_norm=1,
        replace=False,
        mean=True,
        fix_grid=False,
        convex_hull=True,
        **kwargs,
    ):
        super().__init__()
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        self.n_samples = n_samples
        self.fix_grid = fix_grid

        self.grid_size = grid_size
        self.replace = replace
        self.p_norm = p_norm
        self.mean = mean
        self.convex_hull = convex_hull

        self.range_x = torch.linspace(-max_x, max_x, grid_size)
        self.range_y = torch.linspace(-max_y, max_y, grid_size)
        self.range_z = torch.linspace(-max_z, max_z, grid_size)
        self.p = torch.ones(self.range_x.shape)

        self.idx_x = self.p.multinomial(
            num_samples=self.grid_size, replacement=self.replace
        )
        self.idx_y = self.p.multinomial(
            num_samples=self.grid_size, replacement=self.replace
        )
        self.idx_z = self.p.multinomial(
            num_samples=self.grid_size, replacement=self.replace
        )

        range_x = self.range_x[self.idx_x].unsqueeze(dim=-1)
        range_y = self.range_y[self.idx_y].unsqueeze(dim=-1)
        range_z = self.range_z[self.idx_z].unsqueeze(dim=-1)

        self.grid_points = torch.cat([range_x, range_y, range_z], axis=1)

    def batch_pairwise_dist(self, x, y):
        P = torch.cdist(x, y, p=2)
        return P

    def compute_convex_hull(self, gts, preds):
        both = torch.concatenate([gts, preds], dim=1).detach().cpu().numpy()
        min_bound = []
        max_bound = []
        grid_points = []
        for ind in range(len(both)):
            hull = ConvexHull(both[ind])
            if self.convex_hull:
                # sample points within convex hull
                eps = 1e-9
                # check inside and outside for each point
                outside = (
                    np.matmul(
                        hull.equations,
                        np.concatenate(
                            [
                                self.grid_points,
                                np.expand_dims(np.ones(len(self.grid_points)), axis=1),
                            ],
                            axis=1,
                        ).T,
                    )
                    > eps
                ).any(0)
                inside = (
                    np.matmul(
                        hull.equations,
                        np.concatenate(
                            [
                                self.grid_points,
                                np.expand_dims(np.ones(len(self.grid_points)), axis=1),
                            ],
                            axis=1,
                        ).T,
                    )
                    < -eps
                ).all(0)
                both_inside_out = inside + outside

                # get boundary points
                boundary_points = self.grid_points[~both_inside_out]
                boundary_points = torch.tensor(boundary_points).type_as(gts)
                # get inside points
                this_grid = torch.tensor(self.grid_points[inside]).type_as(gts)
                this_grid = torch.concatenate([this_grid, boundary_points], dim=0)

                # upsample to a size
                this_grid = torch.nn.Upsample(size=self.n_samples, mode="linear")(
                    this_grid.T.unsqueeze(dim=0)
                ).T.squeeze()
                grid_points.append(this_grid)
            else:
                # sample points within bounding box cuboid
                min_bound.append(hull.min_bound)
                max_bound.append(hull.max_bound)

                range_x = torch.linspace(
                    hull.min_bound[0], hull.max_bound[0], self.grid_size
                )
                range_y = torch.linspace(
                    hull.min_bound[1], hull.max_bound[1], self.grid_size
                )
                range_z = torch.linspace(
                    hull.min_bound[2], hull.max_bound[2], self.grid_size
                )

                idx_x = self.p.multinomial(
                    num_samples=self.n_samples, replacement=self.replace
                )
                idx_y = self.p.multinomial(
                    num_samples=self.n_samples, replacement=self.replace
                )
                idx_z = self.p.multinomial(
                    num_samples=self.n_samples, replacement=self.replace
                )

                range_x = range_x[idx_x].unsqueeze(dim=-1)
                range_y = range_y[idx_y].unsqueeze(dim=-1)
                range_z = range_z[idx_z].unsqueeze(dim=-1)

                grid_points.append(torch.cat([range_x, range_y, range_z], axis=1))

        grid_points = torch.stack(grid_points, dim=0)
        grid_points = grid_points.type_as(preds)

        return grid_points

    def forward(self, gts, preds):
        grid_points = self.compute_convex_hull(gts, preds)

        P = self.batch_pairwise_dist(grid_points, preds)
        mins, _ = torch.min(P, 2)

        P = self.batch_pairwise_dist(grid_points, gts)
        mins2, _ = torch.min(P, 2)

        if self.p_norm == 1:
            diff = torch.abs(mins - mins2)

        if self.p_norm == 2:
            diff = torch.abs(mins - mins2) ** 2

        if self.mean:
            diff = torch.mean(diff, dim=1)
        else:
            diff = torch.sum(diff, dim=1)

        return diff
