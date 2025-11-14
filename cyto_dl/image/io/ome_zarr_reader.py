from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from monai.config import PathLike
from monai.data import ImageReader
from monai.data.image_reader import _stack_images
from monai.utils import ensure_tuple, require_pkg
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader 
from bioio import BioImage
from bioio_ome_zarr import Reader as BioReader
from upath import UPath as Path
import urllib.parse
import s3fs
import zarr


@require_pkg(pkg_name="upath")
@require_pkg(pkg_name="ome_zarr")
class OmeZarrReader(ImageReader):
    def __init__(self, level=0, image_name="default", channels=None, mip=False):
        super().__init__()
        self.fs = s3fs.S3FileSystem(anon=True)
        self.level = str(level)
        self.image_name = image_name
        self.channels = channels
        self.mip = mip

    def read(self, data: Union[Sequence[PathLike], PathLike]):
        filenames: Sequence[PathLike] = ensure_tuple(data)
        img_ = []
        for path in filenames:
            # if self.image_name:
            #     path = str(Path(path) / self.image_name)

            if 's3' in path:

                if 'https://' not in path:
                    if 'https:/' in path:
                        path = path.replace('https:/','https://')
                    else:
                        raise ValueError(f'file path given {path} is not a url')

                parsed_url = urllib.parse.urlparse(path)
                bucket = parsed_url.netloc.split('.')[0]
                key = parsed_url.path.lstrip('/')
                store = s3fs.S3Map(root=f"{bucket}/{key}", s3=self.fs, check=False)
                # zarr_store = zarr.open(store, mode='r')

                img_.append(store)
            else:
                if 'https://' not in path:
                    if 'https:/' in path:
                        path = path.replace('https:/','https://')
                    else:
                        raise ValueError(f'file path given {path} is not a url')
                img_.append(BioImage(path, reader=BioReader))

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        img_array: List[np.ndarray] = []

        # import pdb; pdb.set_trace()
        # for img_obj in (img):
        try:
            zarr_store = zarr.open(img, mode='r')
            data = zarr_store[self.level][...].astype(np.float32).squeeze()
        except:
            data = img.data.astype(np.float32).squeeze()
        if self.channels:
            if data.ndim > 4:
                data = data[...,self.channels,:,:,:]
            else:
                data = data[self.channels,...]
        if self.mip:
            data = np.max(data, axis=-3)
        img_array.append(data)

        return _stack_images(img_array, {}), {}

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        for fname in ensure_tuple(filename):
            if not str(fname).endswith("zarr"):
                return False
        return True
