import importlib
import io
import os
import uuid
from collections.abc import MutableMapping
from pathlib import Path

import torch
from monai.data.utils import list_data_collate
from omegaconf import OmegaConf
from ts.torch_handler.base_handler import BaseHandler as _BaseHandler
from ts.utils.util import list_classes_from_module

from cyto_dl import utils


class BaseHandler(_BaseHandler):
    """
    Assumptions:
    - hyperparams stored in the .ckpt are final
    - models will be stored as .ckpt
    """

    def __init__(self):
        super().__init__()

        if "config.yaml" not in os.listdir():
            raise FileNotFoundError("`config.yaml` was probably not included in the model archive")

        OmegaConf.register_new_resolver("kv_to_dict", utils.kv_to_dict)
        self.config = OmegaConf.load("config.yaml")

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
        target = self.config.model.pop("_target_")

        return model_class.load_from_checkpoint(model_pt_path, **self.config.model)

    def _load_torchscript_model(self, model_pt_path):
        raise NotImplementedError("We don't support precompiled models (yet?)")

    def preprocess(self, data):
        res = []
        for record in data:
            if isinstance(record, MutableMapping):
                if "body" in record:
                    record = record["body"]

                if "data" in data:
                    record = record["data"]

            res.append(record)
        return list_data_collate(res)

    def inference(self, data):
        with torch.no_grad():
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(self.device)

            loss, preds, _ = self.model.model_step("predict", data, 0)
        return dict(loss=loss, preds=preds)

    def postprocess(self, data):
        mode = self.config["return"].get("mode", "network")

        if mode == "path":
            path = self.config["return"].get("path", "/tmp")  # nosec B108
            response_path = Path(path) / f"{uuid.uuid4()}.pt"
            torch.save(data, response_path)  # nosec B614
            return [str(response_path)]

        buf = io.BytesIO()
        torch.save(data, buf)  # nosec B614
        buf.seek(0)
        return [buf.read()]
