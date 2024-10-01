from pathlib import Path
from typing import Dict, List, Union

import pyrootutils
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from cyto_dl.api.data import ExperimentType
from cyto_dl.eval import evaluate
from cyto_dl.train import train as train_model
from cyto_dl.utils.download_test_data import download_test_data
from cyto_dl.utils.rich_utils import print_config_tree


class CytoDLModel:
    # TODO: add optional CytoDLConfig param to init--if client passes a
    # CytoDLConfig subtype, config will be initialized in constructor and
    # calls to train/predict can be run immediately
    def __init__(self):
        self.cfg = None
        self._training = False
        self._predicting = False

        self.root = pyrootutils.setup_root(
            search_from=__file__,
            project_root_env_var=True,
            dotenv=True,
            pythonpath=True,
            cwd=False,  # do NOT change working directory to root (would cause problems in DDP mode)
            indicator=("pyproject.toml", "README.md"),
        )

    def download_example_data(self):
        download_test_data()

    def load_config_from_file(self, config_path: str):
        """Load configuration file."""
        config_path = Path(config_path)
        assert config_path.exists(), f"config file {config_path} does not exist"
        assert config_path.suffix == ".yaml", f"config file {config_path} must be a yaml file"

        # load config
        self.cfg = OmegaConf.load(config_path)

    def load_config_from_dict(self, config: dict):
        """Load configuration from dictionary."""
        self.cfg = config

    # TODO: replace experiment_type str with api.data.ExperimentType -> will
    # require corresponding changes in ml-segmenter
    def load_default_experiment(
        self, experiment_type: str, output_dir: str, train=True, overrides: List = []
    ):
        """Load configuration from directory."""
        assert experiment_type in {exp_type.value for exp_type in ExperimentType}
        config_dir = self.root / "configs"

        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.2", config_dir=str(config_dir)):
            cfg = compose(
                config_name="train.yaml" if train else "eval.yaml",
                return_hydra_config=True,
                overrides=[f"experiment=im2im/{experiment_type}"] + overrides,
            )

        with open_dict(cfg):
            del cfg["hydra"]
            cfg.extras.enforce_tags = False
            cfg.extras.print_config = False
            cfg["paths"]["output_dir"] = output_dir
            cfg["paths"]["work_dir"] = output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = cfg

    def print_config(self):
        print_config_tree(self.cfg, resolve=True)

    def override_config(self, overrides: Dict[str, Union[str, int, float, bool]]):
        """Override configuration from list of overrides."""
        if self.cfg is None:
            raise ValueError("Configuration must be loaded before overriding!")

        for k, v in overrides.items():
            OmegaConf.update(self.cfg, k, v)

    def save_config(self, path: Path) -> None:
        """Save current config to provided path.

        :param path: path at which to save config
        """
        OmegaConf.save(self.cfg, path)

    async def _train_async(self):
        return train_model(self.cfg)

    async def _predict_async(self):
        return evaluate(self.cfg)

    def train(self, run_async=False, data=None):
        if self.cfg is None:
            raise ValueError("Configuration must be loaded before training!")
        if run_async:
            return self._train_async()
        return train_model(self.cfg, data)

    def predict(self, run_async=False, data=None):
        if self.cfg is None:
            raise ValueError("Configuration must be loaded before predicting!")
        if run_async:
            return self._predict_async()
        return evaluate(self.cfg, data)
