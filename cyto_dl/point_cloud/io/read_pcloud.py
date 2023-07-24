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
        scale: int = 1,
        num_cols: int = 3,
    ):
        """
        Parameters
        ----------
        keys: Union[str, Sequence[str]]
            Key (or list thereof) of the input dictionary to interpret as paths
            to point cloud files which should be loaded
        remote: bool = False
            Whether files can be in a fsspec-interpretable remote location
        """
        super().__init__(keys)
        self.keys = [keys] if isinstance(keys, str) else keys
        self.remote = remote
        self.sample = sample
        self.scale = scale
        self.num_cols = num_cols

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

                res[key] = (
                    torch.tensor(
                        PyntCloud.from_file(path).points.values[:, : self.num_cols],
                        dtype=torch.get_default_dtype(),
                    )
                    * self.scale
                )

                if self.num_cols > 3:
                    res[key][:, self.num_cols - 1 :] = res[key][:, self.num_cols - 1 :] * 0.1

                if self.sample:
                    self.sample_idx = np.random.randint(res[key].shape[0], size=self.sample)
                    res[key] = res[key][self.sample_idx]

        return res
