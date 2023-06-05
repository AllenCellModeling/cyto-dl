import logging

import gorilla
from hydra.utils import instantiate

import aics_im2im.models.handlers.load_image_patch as lip
from aics_im2im.datamodules.dataframe.utils import parse_transforms
from aics_im2im.models.handlers.base_handler import BaseHandler

patches = gorilla.find_patches([lip])
for patch in patches:
    gorilla.apply(patch)

logger = logging.getLogger(__name__)


class ImageHandler(BaseHandler):
    """
    Assumptions:
    - transforms can deal with input being bytes vs a string
    """

    def __init__(self):
        super().__init__()

        transforms = instantiate(self.config.data.transforms)
        self.transforms = parse_transforms(transforms)["predict"]

    def preprocess(self, data):
        return [self.transforms(data) for _ in data]
