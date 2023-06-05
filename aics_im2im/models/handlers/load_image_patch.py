import io
import logging

import gorilla
import monai.transforms.io as monai_io
import torch
from monai.config import NdarrayOrTensor, PathLike
from monai.data.image_reader import ImageReader
from PIL import Image

logger = logging.getLogger(__name__)


@gorilla.patch(monai_io.array.LoadImage, name="__call__")
def load_image__call__patch(
    self,
    img_or_path: bytes | bytearray | NdarrayOrTensor | str | PathLike,
    reader: ImageReader | None = None,
):

    logger.info("Hello, I'm getting used!!!!")

    if isinstance(img_or_path, (str, PathLike)):
        original = gorilla.get_original_attribute(monai_io.array.LoadImage, "__call__")
        return original(self, img_or_path, reader)

    elif isinstance(img_or_path, (bytes, bytearray)):
        return Image.open(io.BytesIO(img_or_path))

    elif isinstance(img_or_path, NdarrayOrTensor):
        return torch.tensor(img_or_path)

    raise TypeError(f"Don't know how to handle `img_or_path` type {type(img_or_path)}")
