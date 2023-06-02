import logging

import torch
from hydra.utils import instantiate

from aics_im2im.datamodules.dataframe.utils import parse_transforms

from .base_handler import BaseHandler

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
