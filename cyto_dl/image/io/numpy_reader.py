import os
import uuid
from tempfile import TemporaryDirectory
from typing import Optional, Sequence, Union

import numpy as np
import torch
from monai.transforms import MapTransform
from upath import UPath as Path


class ReadNumpyFile(MapTransform):
    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        channels: Optional[Sequence[int]] = None,
        remote: bool = False,
        clip_min: Optional[int] = None,
        clip_max: Optional[int] = None,
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
        self.channels = channels
        self.remote = remote
        self.clip_min = clip_min
        self.clip_max = clip_max

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

                res[key] = torch.tensor(np.load(path), dtype=torch.get_default_dtype()).unsqueeze(dim=0)
                if self.channels is not None:
                    res[key] = res[key][self.channels]

                if isinstance(self.clip_min, int):
                    res[key] = torch.where(res[key] > self.clip_min, res[key], 0)

                if isinstance(self.clip_max, int):
                    res[key] = torch.where(res[key] < self.clip_max, res[key], 0)

        return res
