import torch
import torch.nn as nn


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

        self.range_x = torch.linspace(-max_x, max_x, grid_size)
        self.range_y = torch.linspace(-max_y, max_y, grid_size)
        self.range_z = torch.linspace(-max_z, max_z, grid_size)
        self.p = torch.ones(self.range_x.shape)

        self.idx_x = self.p.multinomial(
            num_samples=self.n_samples, replacement=self.replace
        )
        self.idx_y = self.p.multinomial(
            num_samples=self.n_samples, replacement=self.replace
        )
        self.idx_z = self.p.multinomial(
            num_samples=self.n_samples, replacement=self.replace
        )

    def batch_pairwise_dist(self, x, y):
        P = torch.cdist(x, y, p=2)
        return P

    def forward(self, gts, preds):
        bs, _, _ = gts.size()

        if self.fix_grid:
            idx_x = self.idx_x
            idx_y = self.idx_y
            idx_z = self.idx_z
        else:
            idx_x = self.p.multinomial(
                num_samples=self.n_samples, replacement=self.replace
            )
            idx_y = self.p.multinomial(
                num_samples=self.n_samples, replacement=self.replace
            )
            idx_z = self.p.multinomial(
                num_samples=self.n_samples, replacement=self.replace
            )
        range_x = self.range_x[idx_x].unsqueeze(dim=-1)
        range_y = self.range_y[idx_y].unsqueeze(dim=-1)
        range_z = self.range_z[idx_z].unsqueeze(dim=-1)

        grid_points = (
            torch.cat([range_x, range_y, range_z], axis=1)
            .unsqueeze(dim=0)
            .repeat(bs, 1, 1)
        )
        grid_points = grid_points.type_as(preds)

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
