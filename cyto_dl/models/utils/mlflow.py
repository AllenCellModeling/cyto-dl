import logging
import tempfile

import mlflow
from hydra._internal.utils import _locate
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def get_config(tracking_uri, run_id, tmp_dir, mode="train"):
    artifact_path = f"config/{mode}.yaml"
    mlflow.set_tracking_uri(tracking_uri)
    config = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=tmp_dir,
    )

    config = OmegaConf.load(config)
    config = OmegaConf.to_container(config, resolve=True)

    return config


def load_model_from_checkpoint(tracking_uri, run_id, strict=True, path="checkpoints/last.ckpt"):
    mlflow.set_tracking_uri(tracking_uri)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=path, dst_path=tmp_dir
        )

        config = get_config(tracking_uri, run_id, tmp_dir, mode="train")

        model_conf = config["model"]
        model_class = model_conf.pop("_target_")
        model_conf = instantiate(model_conf)
        model_class = _locate(model_class)
        return model_class.load_from_checkpoint(ckpt_path, **model_conf, strict=strict).eval()
