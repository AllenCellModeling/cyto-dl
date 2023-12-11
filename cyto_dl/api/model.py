import pyrootutils
from cyto_dl.train import train
from cyto_dl.eval import evaluate
from pathlib import Path
from typing import List
from omegaconf import OmegaConf, open_dict
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir
from cyto_dl.utils.download_test_data import download_test_data

from cyto_dl.utils.rich_utils import print_config_tree


DEFAULT_EXPERIMENTS=[
    'gan', 'instance_seg', 'labelfree', 'segmentation_plugin','segmentation'
]

class CytoDLModel:
    def __init__(self):
        self.cfg  = None
        self._training = False
        self._predicting = False

        self.root = pyrootutils.setup_root(
            search_from=__file__,
            project_root_env_var=True,
            dotenv=True,
            pythonpath=True,
            cwd=False,  # do NOT change working directory to root (would cause problems in DDP mode)
            indicator= ('pyproject.toml', 'README.md')
        )

    def download_example_data(self):
        download_test_data()

    def load_config_from_file(self, config_path: str):
        """Load configuration file."""
        config_path = Path(config_path)
        assert config_path.exists(), f"config file {config_path} does not exist"
        assert config_path.endswith('.yaml'), f"config file {config_path} must be a yaml file"

        #load config
        self.cfg = OmegaConf.load(config_path)

    def load_config_from_dict(self, config: dict):
        """Load configuration from dictionary."""
        self.cfg = config

    def load_default_experiment(self, experiment_name: str, output_dir:str, train = True, overrides: List = []):
        """Load configuration from directory."""
        assert experiment_name in DEFAULT_EXPERIMENTS
        config_dir = self.root / 'configs'
        
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.2", config_dir=str(config_dir)):
            cfg = compose(config_name='train.yaml' if train else 'eval.yaml', return_hydra_config=True, overrides=[f'experiment=im2im/{experiment_name}'] + overrides)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open_dict(cfg):
            del cfg['hydra']
            cfg.extras.enforce_tags = False
            cfg.extras.print_config = False
            cfg['paths']['output_dir'] = str(output_dir)
            cfg['paths']['work_dir'] = str(output_dir)


        self.cfg = cfg
        OmegaConf.save(self.cfg, output_dir / f'{"train" if train else "eval"}_config.yaml')

    def print_config(self):
        print_config_tree(self.cfg, resolve=True)


    def override_config(self, overrides: List):
        for override in overrides:
            tmp_cfg = self.cfg
            keys, value = override.split('=')
            keys = keys.split('.')
            for key in keys[:-1]:
                tmp_cfg = tmp_cfg.set_default(key, {})
            tmp_cfg[keys[-1]] = value
    
    async def train(self):
        if self.cfg is None:
            raise ValueError('Configuration must be loaded before training!')
        train(self.cfg)

    async def predict(self):
        if self.cfg is None:
            raise ValueError('Configuration must be loaded before predicting!')
        evaluate(self.cfg)

    def get_progress(self):
        print(self.progress)
        self.progress += 1