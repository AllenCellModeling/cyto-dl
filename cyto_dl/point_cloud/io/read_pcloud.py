import os
import uuid
from tempfile import TemporaryDirectory
from typing import Optional, Sequence, Union

import numpy as np
import torch
from monai.transforms import MapTransform
from pyntcloud import PyntCloud
from upath import UPath as Path


class ReadPointCloud(MapTransform):
    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        remote: bool = False,
        sample: Optional[int] = None,
        scale: float = 1.0,
        num_cols: int = 3,
        norm: bool = False,
        flip_dims: bool = False,
        rotate: bool = False,
        scalar_scale: Optional[float] = 0.1,
        final_columns: Optional[list] = None,
    ):
        """
        Parameters
        ----------
        keys: Union[str, Sequence[str]]
            Key (or list thereof) of the input dictionary to interpret as paths
            to point cloud files which should be loaded
        remote: bool = False
            Whether files can be in a fsspec-interpretable remote location
        sample: Optional[int]
            How many points to sample from the point cloud
        scale: float
            scale factor for X,Y,Z coordinates - e.g. X' = scale * X
        num_cols: int
            Number of columns to sample from the saved point cloud
            This is relevant for ply files saved with additional scalar features
            here we assume first 3 columns are X, Y, Z coordinates
        norm: bool
            Whether to normalize point cloud coordinates.
        flip_dims: bool
            Whether to flip dims from XYZ to ZYX
        rotate: bool
            Whether to add random rotation to the point cloud in the XY plane
            assuming ordering is XYZ
        scalar_scale: float
            Scale factor for scalar features

        """
        super().__init__(keys)
        self.keys = [keys] if isinstance(keys, str) else keys
        self.remote = remote
        self.sample = sample
        self.norm = norm
        self.scale = scale
        self.num_cols = num_cols
        self.flip_dims = flip_dims
        self.rotate = rotate
        self.scalar_scale = scalar_scale
        self.final_columns = final_columns

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __call__(self, row):
        res = dict(**row)

        with TemporaryDirectory() as tmp_dir:
            for key in self.keys:
                if self.remote:
                    path = Path(row[key])
                    ext = path.suffix

                    fifo_path = str(Path(tmp_dir) / f"{uuid.uuid4()}{ext}")
                    os.mkfifo(fifo_path)

                    with path.open("rb") as f_input:
                        Path(fifo_path).write_bytes(f_input.read())
                    path = fifo_path
                else:
                    path = str(row[key])
                points = PyntCloud.from_file(path).points

                if "s" in points.columns:
                    points = points[["z", "y", "x", "s"]]
                else:
                    points = points[["z", "y", "x"]]
                points = points.values[:, : self.num_cols]

                if self.rotate:
                    points = rotate_pointcloud(points)

                if self.flip_dims:
                    if self.num_cols == 3:
                        points = points[:, -1::-1].copy()
                    else:
                        points = np.concatenate([points[:, -2::-1], points[:, -1:]], axis=1)

                if self.scale:
                    points = points * self.scale
                if self.norm:
                    points[:, :3] = self.pc_norm(points[:, :3])
                res[key] = torch.tensor(
                    points,
                    dtype=torch.get_default_dtype(),
                )
                if self.num_cols > 3:
                    res[key][:, self.num_cols - 1 :] = (
                        res[key][:, self.num_cols - 1 :] * self.scalar_scale
                    )

                if self.sample:
                    self.sample_idx = np.random.randint(res[key].shape[0], size=self.sample)
                    res[key] = res[key][self.sample_idx]
                if self.final_columns:
                    res[key] = res[key][:, self.final_columns]

        return res


def rotate_pointcloud(pointcloud, rotation_matrix=None, return_rot=False):
    pointcloud_rotated = pointcloud.copy()
    if rotation_matrix is None:
        theta = np.pi * 2 * np.random.choice(24) / 24
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
    pointcloud_rotated[:, [0, 1]] = pointcloud_rotated[:, [0, 1]].dot(
        rotation_matrix
    )  # random rotation (x,y)
    if return_rot:
        return pointcloud_rotated, rotation_matrix, theta

    return pointcloud_rotated
