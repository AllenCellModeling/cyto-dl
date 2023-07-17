import json
from collections.abc import MutableMapping

import gorilla
from hydra.utils import instantiate
from monai.data.utils import list_data_collate

import cyto_dl.models.handlers.load_image_patch as lip
from cyto_dl.datamodules.dataframe.utils import parse_transforms
from cyto_dl.models.handlers.base_handler import BaseHandler


class ImageHandler(BaseHandler):
    """
    Assumptions:
    - transforms can deal with input being bytes vs a string
    """

    def __init__(self):
        super().__init__()

        patches = gorilla.find_patches([lip])
        for patch in patches:
            gorilla.apply(patch)

        transforms = instantiate(self.config.data.transforms)
        self.transforms = parse_transforms(transforms)["predict"]

    def preprocess(self, data):
        res = []
        for record in data:
            if isinstance(record, MutableMapping):
                if "body" in record:
                    record = record["body"]

                if "data" in data:
                    record = record["data"]

            res.append(self.transforms(record))
        return list_data_collate(res)
