from pathlib import Path
from typing import List, Tuple

class ExperimentDataManager:
    """
    For now, this will be based off of experiment output from segmentation_plugin. If necessary,
    we can later make specific data managers for specific experiment types (similar to pattern in
    cyto_dl_model module).
    """
    def __init__(self, experiment_path: Path):
        """
        :param experiment_path: Path to a) an existing experiment directory, b) an empty directory for
        a new experiment, or c) a nonexistent directory for a new experiment
        """
        if not experiment_path.exists():
            experiment_path.mkdir(parents=True)
        elif experiment_path.is_file():
            raise ValueError("expected path to directory, found file instead")

        self._root = experiment_path

    def get_train_output_path(self) -> Path:
        return self._root / "train"
    
    def get_pred_output_path(self) -> Path:
        # create new pred dir in prediction subdir
        pass
    
    def get_test_images_path(self) -> Path:
        test_path: Path = self.get_train_output_path() / "test_images"
        if not test_path.exists():
            raise FileNotFoundError("test image directory does not exist at the expected location")

    def get_model_config_path(self) -> Path:
        cfg_path: Path = self.get_train_output_path() / "train_config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError("config file does not exist at the expected location")
        return cfg_path
    
    def get_best_checkpoint(self) -> Tuple[Path, int]:
        """
        Returns a tuple of (path to best checkpoint, current epoch for that checkpoint)
        """
        checkpoints_path: Path = self.get_train_output_path() / "checkpoints"
        if not checkpoints_path.exists():
            raise FileNotFoundError("checkpoint directory does not exist at the expected location")

        files: List[Path] = [
            entry
            for entry in checkpoints_path.iterdir()
            if entry.is_file() and not "last" in entry.name.lower()
        ]
        if not files:
            raise FileNotFoundError("best checkpoint does not exist at the expected location")

        files.sort(key=lambda file: file.stat().st_mtime)
        best_ckpt: Path = files[-1]
        try:
            epoch: int = int(best_ckpt.name.split(".")[0].split("_")[-1])
        except:
            raise ValueError("best checkpoint filename does not meet formatting requirements")
        
        return best_ckpt, epoch
