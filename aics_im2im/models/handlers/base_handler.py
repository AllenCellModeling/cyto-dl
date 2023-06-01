import importlib
import logging
import os
import pprint
import sys

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from ts.torch_handler.base_handler import BaseHandler as _BaseHandler
from ts.utils.util import list_classes_from_module

from aics_im2im import utils
from aics_im2im.datamodules.dataframe.utils import parse_transforms

logger = logging.getLogger(__name__)


class BaseHandler(_BaseHandler):
    def __init__(self):
        super().__init__()

        if "config.yaml" not in os.listdir():
            raise FileNotFoundError("`config.yaml` was probably not included in the model archive")

        OmegaConf.register_new_resolver("kv_to_dict", utils.kv_to_dict)
        conf = OmegaConf.load("config.yaml")
        transforms = instantiate(conf.data.transforms)
        self.transforms = parse_transforms(transforms)["predict"]

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model_file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError(
                f"Expected only one class as model definition. {model_class_definitions}"
            )

        model_class = model_class_definitions[0]
        return model_class.load_from_checkpoint(model_pt_path)

    def preprocess(self, data):
        logger.info(type(data[0]["data"]))

        return [self.transforms({"image": _["data"]}) for _ in data]
