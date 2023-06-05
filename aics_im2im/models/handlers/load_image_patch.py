import io
import logging
import os
from typing import Optional, Union

import gorilla
import monai.transforms.io as monai_io
import numpy as np
import torch
from monai.data.image_reader import ImageReader
from PIL import Image

logger = logging.getLogger(__name__)

settings = gorilla.Settings(allow_hit=True, store_hit=True)


@gorilla.patch(monai_io.array.LoadImage, name="__call__", settings=settings)
def load_image__call__patch(
    self,
    img_or_path: Union[bytes, bytearray, np.ndarray, torch.Tensor, str, os.PathLike],
    reader: Optional[ImageReader] = None,
):
    if isinstance(img_or_path, (str, os.PathLike)):
        original = gorilla.get_original_attribute(monai_io.array.LoadImage, "__call__")
        return original(self, img_or_path, reader)

    elif isinstance(img_or_path, (bytes, bytearray)):
        out = Image.open(io.BytesIO(img_or_path))

    elif isinstance(img_or_path, (np.ndarray, torch.Tensor)):
        out = torch.tensor(img_or_path)
    else:
        raise TypeError(f"Don't know how to handle `img_or_path` type {type(img_or_path)}")

    if self.image_only:
        return out
    return out, {}
