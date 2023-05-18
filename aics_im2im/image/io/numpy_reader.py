import os
import uuid
from tempfile import TemporaryDirectory
from typing import Sequence, Union, Optional

from monai.transforms import MapTransform
from upath import UPath as Path
import torch
import numpy as np
from torchvision.transforms import Resize
# import torchvision.transforms.functional as F
import torch.nn.functional as F


class ReadNumpyFile(MapTransform):
    def __init__(self, keys: Union[str, Sequence[str]], remote: bool = False, 
    clip_min: Optional[int] = None, clip_max: Optional[int] = None, 
    read_center: Optional[int] = None, resize: Optional[int] = None):
        """
        Parameters
        ----------
        keys: Union[str, Sequence[str]]
            Key (or list thereof) of the input dictionary to interpret as paths
            to point cloud files which should be loaded
        remote: bool = False
            Whether files can be in a fsspec-interpretable remote location
        clip_min, clip_max: Optional - clip values for output
        """
        super().__init__(keys)
        self.keys = [keys] if isinstance(keys, str) else keys
        self.remote = remote
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.read_center = read_center
        self.resize = resize

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

                res[key] = torch.tensor(np.load(path), dtype=torch.get_default_dtype()).unsqueeze(
                    dim=0
                )
                if (self.clip_min is not None) & (self.clip_max is not None):
                    res[key] = np.clip(res[key], self.clip_min, self.clip_max)
                if self.read_center:
                    res[key] = res[key][:,16,...]
                if self.resize:
                    # import ipdb
                    # ipdb.set_trace()
                    res[key] = F.interpolate(res[key].unsqueeze(dim=1), 28)[:,0,...]

        return res
